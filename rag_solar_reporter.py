"""
rag_solar_reporter.py
=====================
RAG-Powered Solar Scientist — Upgrades ai_reporter.py with Physics Grounding
Mission: ISRO Aditya-L1

CURRENT LIMITATION OF ai_reporter.py
──────────────────────────────────────
Your current Gemini reporter describes what it sees:
  "Bright region detected in the Ca II H band with elevated MSE score."

This is correct but not scientifically useful to ISRO. What they need:
  "This morphology closely resembles the 2024-09-07 X2.1 flare (AR 13825),
   which produced a 20-minute HF radio blackout over Asia-Pacific.
   The Mg II k/h ratio (1.02) is consistent with class M-X transition
   as described in Druett et al. 2023 (A&A, 672, A85).
   Recommend immediate GOES alert correlation check."

WHAT RAG ADDS
──────────────
RAG = Retrieval-Augmented Generation

1. BUILD: Index a vector database of solar physics papers + ISRO SUIT docs
   - NASA ADS API → fetch abstracts + key sections of relevant papers
   - GOES event catalog → historical flare descriptions
   - ISRO SUIT observation logs
   Sources → chunks → embeddings → ChromaDB vector store

2. RETRIEVE: At anomaly time, embed the current situation description
   and find the 5 most similar historical cases in the database
   "This looks like..." → semantic search → returns similar past events

3. GENERATE: Gemini gets the current observation + 5 retrieved contexts
   → Produces physically grounded, citation-backed PDF report

ARCHITECTURE
────────────
  SolarRAGPipeline:
    ├── DocumentIngester    — downloads and processes papers/catalogs
    ├── VectorStore         — ChromaDB with sentence-transformer embeddings
    ├── SituationBuilder    — converts anomaly_status.json + images → query
    ├── GeminiRagReporter   — calls Gemini with retrieved context
    └── PDFBuilder          — formats the grounded report (replaces fpdf usage)
"""

import os
import json
import time
import logging
import requests
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import numpy as np

# ---------------------------------------------------------------------------
# Dynamic Directory Setup
# ---------------------------------------------------------------------------
PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR    = os.path.join(PROJECT_DIR, "reports")
LOG_DIR        = os.path.join(PROJECT_DIR, "logs")
DATA_DIR       = os.path.join(PROJECT_DIR, "data")
RAG_DIR        = os.path.join(DATA_DIR, "rag_knowledge_base")
CHROMADB_DIR   = os.path.join(RAG_DIR, "chromadb")
PAPERS_DIR     = os.path.join(RAG_DIR, "papers")
STATUS_JSON    = os.path.join(LOG_DIR,  "anomaly_status.json")
GOES_STATUS    = os.path.join(LOG_DIR,  "goes_status.json")

for d in [REPORTS_DIR, LOG_DIR, RAG_DIR, CHROMADB_DIR, PAPERS_DIR]:
    os.makedirs(d, exist_ok=True)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document Ingester — fetches solar physics papers and event catalogs
# ---------------------------------------------------------------------------

class SolarDocumentIngester:
    """
    Fetches and processes solar physics documents for the RAG knowledge base.

    Sources:
      1. NASA ADS API — solar physics paper abstracts + key sections
      2. NOAA SWPC GOES flare catalog — historical event descriptions
      3. SUIT/ISRO observation reports (if available locally)
    """

    NASA_ADS_API = "https://api.adsabs.harvard.edu/v1/search/query"
    GOES_EVENTS  = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"

    # Curated list of highly relevant solar flare papers
    SEED_QUERIES = [
        "SUIT solar ultraviolet imaging telescope Aditya-L1 flare",
        "solar flare Mg II chromosphere UV ribbon",
        "Ca II H solar flare chromospheric evaporation",
        "solar flare classification GOES X-ray UV correlation",
        "CME coronal mass ejection optical flow velocity field",
        "solar active region emergence magnetic flux flare prediction",
        "solar flare wavelet analysis high frequency brightening",
        "quiet sun chromosphere Mg II line core intensity",
        "solar UV 2796 angstrom Mg II k flare impulsive phase",
        "deep learning solar flare anomaly detection autoencoder",
    ]

    def __init__(
        self,
        ads_token:    Optional[str] = None,
        max_papers:   int           = 200,
        chunk_size:   int           = 512,     # characters per chunk
        chunk_overlap: int          = 64,
    ):
        self.ads_token    = ads_token or os.environ.get("ADS_API_TOKEN", "")
        self.max_papers   = max_papers
        self.chunk_size   = chunk_size
        self.chunk_overlap = chunk_overlap

    def fetch_ads_papers(self, query: str, rows: int = 20) -> List[Dict]:
        """Fetch paper metadata from NASA ADS."""
        if not self.ads_token:
            log.warning("[RAG] No ADS API token. Get one at https://ui.adsabs.harvard.edu/user/settings/token")
            return []

        params = {
            "q":  query,
            "fl": "title,author,abstract,year,bibcode,doi",
            "rows": rows,
            "sort": "citation_count desc",
        }
        headers = {"Authorization": f"Bearer {self.ads_token}"}

        try:
            r = requests.get(self.NASA_ADS_API, params=params,
                             headers=headers, timeout=15)
            r.raise_for_status()
            return r.json().get("response", {}).get("docs", [])
        except Exception as e:
            log.warning(f"[RAG] ADS fetch failed for '{query}': {e}")
            return []

    def fetch_goes_events(self) -> List[Dict]:
        """Fetch recent GOES flare events as knowledge base entries."""
        try:
            r = requests.get(self.GOES_EVENTS, timeout=15)
            r.raise_for_status()
            data = r.json()
            # Summarise into text entries
            entries = []
            prev_flux = 0
            for rec in data:
                if rec.get("energy") != "0.1-0.8nm":
                    continue
                flux = float(rec.get("flux", 0))
                if flux > 1e-6 and flux > prev_flux * 2:  # C-class or above, rising
                    cls_map = {1e-4: "X", 1e-5: "M", 1e-6: "C", 1e-7: "B"}
                    cls = next((v for k, v in cls_map.items() if flux >= k), "A")
                    entries.append({
                        "source": "GOES_event_catalog",
                        "text": (
                            f"GOES event detected at {rec['time_tag']}. "
                            f"XRS-B flux: {flux:.2e} W/m² ({cls}-class). "
                            f"This indicates {cls}-class solar flare activity. "
                            f"X-class flares cause major HF radio blackouts and "
                            f"radiation storms. M-class cause minor HF disruptions. "
                        ),
                        "metadata": {
                            "type": "goes_event",
                            "flux": flux,
                            "class": cls,
                            "time": rec["time_tag"],
                        }
                    })
                prev_flux = max(prev_flux, flux)
            return entries[:50]   # Top 50 events
        except Exception as e:
            log.warning(f"[RAG] GOES fetch failed: {e}")
            return []

    def paper_to_chunks(self, paper: Dict) -> List[Dict]:
        """Convert a paper's abstract + metadata into overlapping text chunks."""
        title    = paper.get("title", [""])[0] if isinstance(paper.get("title"), list) else paper.get("title", "")
        abstract = paper.get("abstract", "")
        authors  = ", ".join(paper.get("author", [])[:3])
        year     = paper.get("year", "")
        bibcode  = paper.get("bibcode", "")
        doi      = paper.get("doi", [""])[0] if paper.get("doi") else ""

        full_text = (
            f"Title: {title}\n"
            f"Authors: {authors} ({year})\n"
            f"DOI: {doi}\n"
            f"Abstract: {abstract}"
        )

        # Split into overlapping chunks
        chunks  = []
        text    = full_text
        start   = 0
        chunk_n = 0
        while start < len(text):
            end   = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            if len(chunk.strip()) > 50:  # Skip tiny chunks
                chunks.append({
                    "text":     chunk,
                    "source":   "nasa_ads",
                    "metadata": {
                        "title":   title[:100],
                        "authors": authors,
                        "year":    year,
                        "bibcode": bibcode,
                        "doi":     doi,
                        "chunk":   chunk_n,
                    }
                })
            start  += self.chunk_size - self.chunk_overlap
            chunk_n += 1

        return chunks

    def ingest_all(self) -> List[Dict]:
        """Fetch all sources and return list of text chunks for indexing."""
        all_chunks = []

        # NASA ADS papers
        seen_bibcodes = set()
        for query in self.SEED_QUERIES:
            papers = self.fetch_ads_papers(query, rows=20)
            for paper in papers:
                bc = paper.get("bibcode", "")
                if bc in seen_bibcodes:
                    continue
                seen_bibcodes.add(bc)
                all_chunks.extend(self.paper_to_chunks(paper))
                if len(seen_bibcodes) >= self.max_papers:
                    break
            if len(seen_bibcodes) >= self.max_papers:
                break

        # GOES event catalog
        all_chunks.extend(self.fetch_goes_events())

        # Built-in solar physics knowledge (fallback when no ADS token)
        all_chunks.extend(self._builtin_knowledge())

        log.info(f"[RAG] Ingested {len(all_chunks)} document chunks")
        return all_chunks

    def _builtin_knowledge(self) -> List[Dict]:
        """Hard-coded solar physics facts — used when ADS token not available."""
        facts = [
            "Solar flares are classified by their GOES X-ray peak flux: A (<1e-7), B (1e-7 to 1e-6), C (1e-6 to 1e-5), M (1e-5 to 1e-4), X (>1e-4 W/m²). X-class flares are the most energetic and can cause severe geomagnetic storms.",
            "The Mg II k spectral line at 2796Å is a chromospheric temperature diagnostic. During solar flares, intense brightening in this line indicates chromospheric heating to temperatures above 10,000K.",
            "The Ca II H line at 3968Å forms in the upper chromosphere and is one of the most sensitive optical indicators of solar flare activity. Flare ribbons appear as bright features separating from the magnetic polarity inversion line.",
            "SUIT (Solar Ultraviolet Imaging Telescope) on Aditya-L1 operates in 11 filter bands from 2000-4000Å, covering photospheric and chromospheric emission. The NB3 (2796Å) and NB8 (3968Å) bands are primary flare detection channels.",
            "Coronal Mass Ejections (CMEs) are large eruptions of plasma and magnetic field from the Sun. They travel at 200-3000 km/s and can be detected as optical flow signatures in UV chromospheric imaging. CMEs that reach Earth can cause geomagnetic storms rated G1-G5.",
            "Pre-flare magnetic flux emergence typically precedes major flares by 12-48 hours. Observable signatures include increased chromospheric brightenings, UV brightening at polarity inversion lines, and changes in Ca II H line-of-sight magnetograms.",
            "The Aditya-L1 spacecraft orbits the L1 Lagrange point, providing uninterrupted solar observation. Its SUIT payload captures 560×560 pixel images with 0.7 arcsec/pixel resolution, covering active regions with full-disk capability.",
            "Solar Cycle 25 began in December 2019 and is currently near solar maximum (predicted 2025). Solar maximum is characterised by increased sunspot numbers, higher flare frequency, and more frequent X-class events.",
            "Optical flow analysis between consecutive UV solar images can detect CME eruption onset. The Farneback algorithm applied to Ca II H or Mg II images shows radially outward velocity fields during eruptions.",
            "The reconstruction error of a Variational Autoencoder trained on quiet-sun images provides an unsupervised anomaly score. High MSE in Ca II H or Mg II bands correlates with flare events, as confirmed by Huang et al. 2023 (ApJ, 945, 38).",
        ]
        return [
            {
                "text":     fact,
                "source":   "builtin_knowledge",
                "metadata": {"type": "physics_fact", "index": i}
            }
            for i, fact in enumerate(facts)
        ]


# ---------------------------------------------------------------------------
# Vector Store (ChromaDB)
# ---------------------------------------------------------------------------

class SolarVectorStore:
    """
    ChromaDB-based vector store for solar physics documents.
    Uses sentence-transformers for embedding (all-MiniLM-L6-v2).
    """

    EMBED_MODEL = "all-MiniLM-L6-v2"   # Lightweight, fast, good semantic retrieval

    def __init__(self, persist_dir: str = CHROMADB_DIR):
        self.persist_dir = persist_dir
        self._client     = None
        self._collection = None
        self._embedder   = None

    def _init_chromadb(self):
        """Lazy init — only import when actually needed."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                self._client = chromadb.PersistentClient(
                    path=self.persist_dir,
                    settings=Settings(anonymized_telemetry=False)
                )
                self._collection = self._client.get_or_create_collection(
                    name="solar_physics",
                    metadata={"hnsw:space": "cosine"}
                )
                log.info(f"[RAG] ChromaDB ready at {self.persist_dir}")
            except ImportError:
                raise ImportError(
                    "chromadb not installed. Run: pip install chromadb sentence-transformers"
                )

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.EMBED_MODEL)
        return self._embedder

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._get_embedder().encode(texts, show_progress_bar=False).tolist()

    def index(self, chunks: List[Dict]) -> None:
        """Index document chunks into ChromaDB."""
        self._init_chromadb()

        # Batch by 100 to avoid memory issues
        batch_size = 100
        total      = 0
        for i in range(0, len(chunks), batch_size):
            batch    = chunks[i:i+batch_size]
            texts    = [c["text"] for c in batch]
            embeddings = self.embed(texts)
            ids      = [hashlib.md5(t.encode()).hexdigest() for t in texts]
            metadatas = [c.get("metadata", {}) for c in batch]

            self._collection.upsert(
                documents  = texts,
                embeddings = embeddings,
                ids        = ids,
                metadatas  = metadatas,
            )
            total += len(batch)
            if total % 500 == 0:
                log.info(f"[RAG] Indexed {total}/{len(chunks)} chunks")

        log.info(f"[RAG] Total indexed: {self._collection.count()} chunks")

    def query(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """Retrieve most relevant chunks for a query."""
        self._init_chromadb()
        query_emb = self.embed([query_text])
        results   = self._collection.query(
            query_embeddings = query_emb,
            n_results        = n_results,
            include          = ["documents", "metadatas", "distances"],
        )
        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append({
                "text":       doc,
                "metadata":   meta,
                "similarity": 1 - dist,   # cosine: 1=identical, 0=unrelated
            })
        return retrieved

    @property
    def count(self) -> int:
        self._init_chromadb()
        return self._collection.count()


# ---------------------------------------------------------------------------
# Situation Builder — converts anomaly state → natural language query
# ---------------------------------------------------------------------------

class SituationBuilder:
    """
    Reads anomaly_status.json, goes_status.json, and recent EDA graphs
    and builds a natural language description of the current solar situation
    for RAG retrieval.
    """

    def build_query(self) -> str:
        """Build retrieval query from current system state."""
        parts = []

        # Anomaly detector state
        if os.path.exists(STATUS_JSON):
            try:
                with open(STATUS_JSON) as f:
                    status = json.load(f)
                confidence  = status.get("confidence", "NONE")
                vae_score   = status.get("vae_score", 0.0)
                iso_score   = status.get("iso_score", 0.0)
                n_triggered = status.get("n_triggered", 0)

                parts.append(
                    f"Solar anomaly detection: confidence={confidence}, "
                    f"n_detectors_triggered={n_triggered}, "
                    f"VAE_reconstruction_error={vae_score:.4f}, "
                    f"telemetry_anomaly_score={iso_score:.4f}."
                )
            except Exception:
                pass

        # GOES state
        if os.path.exists(GOES_STATUS):
            try:
                with open(GOES_STATUS) as f:
                    goes = json.load(f)
                flux  = goes.get("latest_xrsb_flux", 0)
                cls   = goes.get("latest_class", "A")
                active = goes.get("active_flare", False)
                n_ev  = goes.get("event_count_6h", 0)
                parts.append(
                    f"GOES X-ray status: current_class={cls}, "
                    f"flux={flux:.2e}_W/m², active_flare={active}, "
                    f"events_last_6h={n_ev}."
                )
            except Exception:
                pass

        # Fallback
        if not parts:
            parts.append("Solar observation anomaly detected in SUIT UV imaging data.")

        query = " ".join(parts)
        query += " Related solar flare signatures: Mg II k 2796 angstrom brightening, Ca II H flare ribbon, chromospheric evaporation."
        return query

    def build_context_string(self, retrieved: List[Dict]) -> str:
        """Format retrieved documents as a context block for Gemini."""
        lines = ["=== RETRIEVED SOLAR PHYSICS CONTEXT ===\n"]
        for i, doc in enumerate(retrieved, 1):
            meta = doc.get("metadata", {})
            sim  = doc.get("similarity", 0)
            src  = meta.get("title", meta.get("type", "solar_physics_document"))
            lines.append(
                f"[Reference {i}] Similarity: {sim:.3f} | Source: {src}\n"
                f"{doc['text']}\n"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RAG-Powered Gemini Reporter
# ---------------------------------------------------------------------------

class SolarRAGReporter:
    """
    Upgrades ai_reporter.py's Gemini call with RAG-retrieved solar physics context.

    Drop-in replacement:
        Old: call Gemini with just the image + anomaly_status.json
        New: call Gemini with image + anomaly_status.json + 5 retrieved papers

    The resulting PDF report contains:
      1. Current observation summary
      2. Detector status (VAE, IsoForest, OCSVM, GOES)
      3. Retrieved similar historical events with citations
      4. AI-generated analysis grounded in solar physics literature
      5. Recommended actions (based on similarity to known events)
    """

    GEMINI_MODEL = "gemini-2.5-flash-preview-05-14"

    def __init__(
        self,
        vector_store:  SolarVectorStore,
        gemini_api_key: Optional[str] = None,
        n_retrieved:   int  = 5,
        temperature:   float = 0.1,  # Low temp for scientific accuracy
    ):
        self.store       = vector_store
        self.api_key     = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        self.n_retrieved = n_retrieved
        self.temperature = temperature
        self.builder     = SituationBuilder()

    def generate_report(
        self,
        solar_image_path:  Optional[str] = None,
        eda_plot_paths:    Optional[List[str]] = None,
        output_path:       Optional[str] = None,
    ) -> str:
        """
        Generate a RAG-grounded PDF report for the current solar observation.

        Returns the path to the generated PDF.
        """
        if output_path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            output_path = os.path.join(REPORTS_DIR, f"solar_report_{ts}.pdf")

        # 1. Build query from current system state
        query   = self.builder.build_query()
        log.info(f"[RAG] Query: {query[:120]}...")

        # 2. Retrieve relevant papers/events
        retrieved = self.store.query(query, n_results=self.n_retrieved)
        context   = self.builder.build_context_string(retrieved)
        log.info(f"[RAG] Retrieved {len(retrieved)} documents "
                 f"(top similarity: {retrieved[0]['similarity']:.3f})")

        # 3. Build Gemini prompt with retrieved context
        prompt = self._build_prompt(query, context)

        # 4. Call Gemini API
        report_text = self._call_gemini(prompt, solar_image_path)

        # 5. Build PDF
        self._build_pdf(report_text, retrieved, output_path)

        log.info(f"[RAG] Report saved: {output_path}")
        return output_path

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""You are an expert solar physicist analysing data from ISRO's Aditya-L1 SUIT instrument.

CURRENT OBSERVATION SITUATION:
{query}

{context}

Using the retrieved context above as scientific grounding, generate a comprehensive solar weather report that includes:

1. EXECUTIVE SUMMARY (2-3 sentences for mission operators)
2. DETECTOR STATUS — Report each anomaly detector's findings with its score
3. PHYSICAL INTERPRETATION — Based on the retrieved papers, what physical process is likely occurring?
4. HISTORICAL COMPARISON — Which retrieved reference events are most similar? Cite specifically.
5. THREAT ASSESSMENT — Rate: ROUTINE / ELEVATED / WARNING / EMERGENCY with justification
6. RECOMMENDED ACTIONS — Specific steps for ISRO mission operations
7. SCIENTIFIC NOTES — Any notable observations for solar physics analysis

Be precise, cite the retrieved references by number [Ref N], and ground all claims in solar physics.
If GOES shows active flaring, correlate with SUIT UV observations explicitly."""

    def _call_gemini(self, prompt: str,
                     image_path: Optional[str] = None) -> str:
        """Call Gemini API with optional image."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.GEMINI_MODEL)

            content = [prompt]
            if image_path and os.path.exists(image_path):
                img = genai.upload_file(image_path)
                content = [img, prompt]

            resp = model.generate_content(
                content,
                generation_config=genai.GenerationConfig(temperature=self.temperature)
            )
            return resp.text
        except Exception as e:
            log.error(f"[RAG] Gemini call failed: {e}")
            return f"[Report generation failed: {e}]\n\nQuery was:\n{prompt[:500]}"

    def _build_pdf(self, report_text: str, retrieved: List[Dict],
                   output_path: str) -> None:
        """Build formatted PDF using fpdf2."""
        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)

            # Header
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Aditya-L1 SUIT — Solar Weather Intelligence Report", ln=True)
            pdf.set_font("Helvetica", size=10)
            pdf.cell(0, 6, f"Generated: {datetime.now(timezone.utc).isoformat()}", ln=True)
            pdf.ln(5)

            # RAG grounded report
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "AI Analysis (RAG-Grounded)", ln=True)
            pdf.set_font("Helvetica", size=10)
            for line in report_text.split("\n"):
                clean = line.encode("ascii", errors="replace").decode("ascii")
                pdf.multi_cell(0, 5, clean)
            pdf.ln(5)

            # References section
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Retrieved References", ln=True)
            pdf.set_font("Helvetica", size=9)
            for i, doc in enumerate(retrieved, 1):
                meta = doc.get("metadata", {})
                ref  = f"[{i}] {meta.get('title', 'Solar physics document')} "
                ref += f"({meta.get('authors', '')}, {meta.get('year', '')})"
                pdf.multi_cell(0, 4, ref.encode("ascii", errors="replace").decode("ascii"))

            pdf.output(output_path)
        except ImportError:
            # Fallback: plain text report
            with open(output_path.replace(".pdf", ".txt"), "w") as f:
                f.write(report_text)
            log.warning("[RAG] fpdf2 not installed — saved as .txt instead")


# ---------------------------------------------------------------------------
# Main Pipeline Class
# ---------------------------------------------------------------------------

class SolarRAGPipeline:
    """
    Complete RAG pipeline — build knowledge base and generate reports.

    Usage:
        pipeline = SolarRAGPipeline(ads_token="...", gemini_key="...")

        # First run: build knowledge base (takes ~5 min)
        pipeline.build_knowledge_base()

        # Ongoing: generate reports when anomaly detected
        pipeline.generate_report(solar_image_path="path/to/latest.png")
    """

    def __init__(
        self,
        ads_token:   Optional[str] = None,
        gemini_key:  Optional[str] = None,
        rebuild:     bool          = False,
    ):
        self.ingester = SolarDocumentIngester(ads_token=ads_token)
        self.store    = SolarVectorStore(CHROMADB_DIR)
        self.reporter = SolarRAGReporter(self.store, gemini_key)
        self.rebuild  = rebuild

    def build_knowledge_base(self) -> None:
        """Ingest documents into the vector store. Run once."""
        if not self.rebuild and self.store.count > 100:
            log.info(f"[RAG] Knowledge base exists ({self.store.count} docs). "
                     f"Pass rebuild=True to force rebuild.")
            return

        log.info("[RAG] Building solar physics knowledge base...")
        chunks = self.ingester.ingest_all()
        self.store.index(chunks)
        log.info(f"[RAG] Knowledge base ready: {self.store.count} chunks")

    def generate_report(
        self,
        solar_image_path: Optional[str] = None,
        eda_plots:        Optional[List[str]] = None,
    ) -> str:
        return self.reporter.generate_report(solar_image_path, eda_plots)

    def run_daemon(self, check_interval: int = 300) -> None:
        """
        Run as a daemon — generate report whenever HIGH or MEDIUM anomaly detected.
        Replaces/augments the existing ai_reporter.py daemon.
        """
        log.info(f"[RAG] Reporter daemon started (check every {check_interval}s)")
        last_report_time = 0

        while True:
            try:
                if os.path.exists(STATUS_JSON):
                    with open(STATUS_JSON) as f:
                        status = json.load(f)

                    confidence = status.get("confidence", "NONE")
                    if confidence in ["HIGH", "MEDIUM"]:
                        now = time.time()
                        if now - last_report_time > 600:   # Min 10 min between reports
                            log.info(f"[RAG] {confidence} anomaly — generating report...")
                            self.generate_report()
                            last_report_time = now
            except Exception as e:
                log.error(f"[RAG] Daemon error: {e}")

            time.sleep(check_interval)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser(description="RAG Solar Reporter")
    p.add_argument("--build",    action="store_true", help="Build knowledge base")
    p.add_argument("--report",   action="store_true", help="Generate one report now")
    p.add_argument("--daemon",   action="store_true", help="Run as background daemon")
    p.add_argument("--rebuild",  action="store_true", help="Force rebuild knowledge base")
    p.add_argument("--ads_token",  default=None)
    p.add_argument("--gemini_key", default=None)
    p.add_argument("--image",    default=None, help="Solar image path for report")
    args = p.parse_args()

    pipeline = SolarRAGPipeline(
        ads_token  = args.ads_token,
        gemini_key = args.gemini_key,
        rebuild    = args.rebuild,
    )

    if args.build or args.daemon or args.report:
        pipeline.build_knowledge_base()

    if args.report:
        path = pipeline.generate_report(args.image)
        print(f"Report saved: {path}")

    if args.daemon:
        pipeline.run_daemon()
