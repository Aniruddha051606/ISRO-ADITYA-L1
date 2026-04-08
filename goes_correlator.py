"""
goes_correlator.py
==================
GOES-16/17 X-ray Flux Correlation Daemon — Ground Truth Validator
Mission: ISRO Aditya-L1

WHY GOES CORRELATION?
──────────────────────
NOAA's GOES-16 and GOES-17 satellites carry the EXIS instrument which
measures solar X-ray flux in two bands:
  • 0.05–0.4 nm  (short wavelength, XRS-A) — captures impulsive hard X-rays
  • 0.1–0.8 nm   (long wavelength,  XRS-B) — captures soft X-ray thermal emission

This is the GOLD STANDARD for solar flare classification:
  A-class:  10⁻⁸  W/m²  (microflares, barely detectable)
  B-class:  10⁻⁷  W/m²
  C-class:  10⁻⁶  W/m²  (common, minor effects)
  M-class:  10⁻⁵  W/m²  (moderate, some HF radio blackouts)
  X-class:  10⁻⁴  W/m²  (major flares, SUIT will saturate)

HOW IT INTEGRATES:
  1. This daemon polls NOAA's GOES JSON endpoint every 60 seconds
  2. Downloads the latest 6-hour X-ray flux time-series
  3. Detects flux rises (derivative > threshold) → generates GOES alerts
  4. Cross-correlates GOES alert timestamps with anomaly_status.json
     from our ensemble detector to measure TRUE POSITIVE / FALSE POSITIVE rates
  5. Writes a correlation_report.json used by ai_reporter.py for the PDF

NOAA DATA SOURCE (free, public):
  https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json
  Updated every 1 minute. No authentication required.

FLARE CLASSIFICATION EMITTED TO:
  logs/goes_status.json   — consumed by mission_control.py for the 🚨 banner
  reports/goes_correlation.json — consumed by ai_reporter.py
"""

import os
import json
import time
import logging
import requests
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Tuple

# ---------------------------------------------------------------------------
# Dynamic Directory Setup
# ---------------------------------------------------------------------------
PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR        = os.path.join(PROJECT_DIR, "logs")
REPORTS_DIR    = os.path.join(PROJECT_DIR, "reports")
DATA_DIR       = os.path.join(PROJECT_DIR, "data")
GOES_CACHE_DIR = os.path.join(DATA_DIR, "goes_cache")

os.makedirs(LOG_DIR,        exist_ok=True)
os.makedirs(REPORTS_DIR,    exist_ok=True)
os.makedirs(GOES_CACHE_DIR, exist_ok=True)

# Default file paths (can be overridden via CLI args)
DEFAULT_STATUS_PATH      = os.path.join(LOG_DIR,     "goes_status.json")
DEFAULT_ANOMALY_LOG_PATH = os.path.join(LOG_DIR,     "anomaly_log.jsonl")
DEFAULT_CORR_PATH        = os.path.join(REPORTS_DIR, "goes_correlation.json")

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NOAA GOES Fetcher
# ---------------------------------------------------------------------------

GOES_PRIMARY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
GOES_BACKUP_URL  = "https://services.swpc.noaa.gov/json/goes/secondary/xrays-6-hour.json"


class GOESFetcher:
    """
    Downloads the latest GOES X-ray flux data from NOAA SWPC.
    Automatically falls back to the secondary satellite if primary fails.
    """

    def __init__(self, timeout: int = 15, cache_dir: str = GOES_CACHE_DIR):
        self.timeout   = timeout
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self) -> Optional[List[Dict]]:
        """
        Returns a list of dicts:
            [{"time_tag": "2024-01-01T00:00:00Z",
              "satellite": 16,
              "energy": "0.1-0.8nm",
              "flux": 1.2e-7}, ...]
        """
        for url in [GOES_PRIMARY_URL, GOES_BACKUP_URL]:
            try:
                r = requests.get(url, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                self._cache(data)
                log.info(f"[GOES] Fetched {len(data)} records from {url.split('/')[6]}")
                return data
            except Exception as e:
                log.warning(f"[GOES] Failed to fetch from {url}: {e}")

        # Fallback: load latest cache
        return self._load_cache()

    def _cache(self, data: List[Dict]) -> None:
        path = self.cache_dir / "goes_latest.json"
        with open(path, "w") as f:
            json.dump(data, f)

    def _load_cache(self) -> Optional[List[Dict]]:
        path = self.cache_dir / "goes_latest.json"
        if path.exists():
            log.info("[GOES] Using cached data.")
            with open(path) as f:
                return json.load(f)
        return None


# ---------------------------------------------------------------------------
# GOES Data Parser
# ---------------------------------------------------------------------------

class GOESParser:
    """
    Parses raw GOES JSON into structured numpy arrays for analysis.
    Filters to the XRS-B (0.1–0.8 nm) long-wavelength channel, which
    is the standard flare classification channel.
    """

    XRSB_ENERGY = "0.1-0.8nm"
    XRSA_ENERGY = "0.05-0.4nm"

    def parse(self, data: List[Dict], channel: str = "B") -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            data    : Raw GOES JSON list
            channel : "A" (hard X-ray) or "B" (soft X-ray, standard)
        Returns:
            timestamps : (N,) float64 — Unix timestamps
            flux       : (N,) float64 — W/m² flux values
        """
        energy_str = self.XRSB_ENERGY if channel == "B" else self.XRSA_ENERGY

        times, fluxes = [], []
        for record in data:
            if record.get("energy") != energy_str:
                continue
            try:
                t = datetime.fromisoformat(
                    record["time_tag"].replace("Z", "+00:00")
                ).timestamp()
                f = float(record["flux"])
                if f > 0:
                    times.append(t)
                    fluxes.append(f)
            except (ValueError, KeyError):
                continue

        return np.array(times), np.array(fluxes)

    @staticmethod
    def classify(flux_value: float) -> str:
        """Classify a flux value into GOES flare class."""
        if flux_value >= 1e-4: return "X"
        if flux_value >= 1e-5: return "M"
        if flux_value >= 1e-6: return "C"
        if flux_value >= 1e-7: return "B"
        return "A"

    @staticmethod
    def subclass(flux_value: float) -> str:
        """Returns full GOES class string e.g. 'M2.3'."""
        letter = GOESParser.classify(flux_value)
        thresholds = {"X": 1e-4, "M": 1e-5, "C": 1e-6, "B": 1e-7, "A": 1e-8}
        base = thresholds[letter]
        number = flux_value / base
        return f"{letter}{number:.1f}"


# ---------------------------------------------------------------------------
# Flare Event Detector (on the GOES time series)
# ---------------------------------------------------------------------------

class GOESFlareDetector:
    """
    Detects flare onset events from the GOES XRS-B time series using
    a derivative-based trigger (the same method used by NOAA's automated system).

    Algorithm:
      1. Compute dFlux/dt over a sliding window
      2. Trigger when flux rises by > rise_threshold_factor × background in < rise_window_s
      3. Peak = maximum flux in the [trigger, trigger + peak_window_s] interval
      4. End = when flux drops back to < 1.5 × pre-flare background
    """

    def __init__(
        self,
        rise_threshold_factor: float = 2.0,   # Flux must double within rise_window_s
        rise_window_s:         int   = 300,    # 5-minute rise window
        min_class:             str   = "B",    # Minimum class to report
    ):
        self.rise_factor  = rise_threshold_factor
        self.rise_window  = rise_window_s
        self.min_flux     = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}[min_class]
        self.parser       = GOESParser()

    def detect(
        self,
        timestamps: np.ndarray,
        flux:       np.ndarray,
    ) -> List[Dict]:
        """
        Returns list of detected flare events, each as:
            {
                "onset_time": ISO8601 string,
                "peak_time":  ISO8601 string,
                "peak_flux":  float (W/m²),
                "class":      "M2.3",
                "duration_s": int,
            }
        """
        if len(flux) < 10:
            return []

        events = []
        in_flare = False
        onset_idx = None

        # Background: rolling 10-minute median
        bg_window = 10
        bg = np.array([
            np.median(flux[max(0, i-bg_window):i+1])
            for i in range(len(flux))
        ])
        bg = np.maximum(bg, 1e-9)   # Floor to avoid division by zero

        for i in range(bg_window, len(flux) - 1):
            ratio = flux[i] / bg[i]
            if not in_flare and ratio >= self.rise_factor and flux[i] >= self.min_flux:
                in_flare  = True
                onset_idx = i
            elif in_flare:
                # Check if flux has returned to < 1.5× background
                if flux[i] < 1.5 * bg[onset_idx]:
                    peak_idx  = onset_idx + int(np.argmax(flux[onset_idx:i+1]))
                    peak_flux = float(flux[peak_idx])
                    event = {
                        "onset_time": datetime.fromtimestamp(
                            float(timestamps[onset_idx]), tz=timezone.utc
                        ).isoformat(),
                        "peak_time": datetime.fromtimestamp(
                            float(timestamps[peak_idx]), tz=timezone.utc
                        ).isoformat(),
                        "peak_flux":  peak_flux,
                        "class":      self.parser.subclass(peak_flux),
                        "duration_s": int(timestamps[i] - timestamps[onset_idx]),
                    }
                    events.append(event)
                    in_flare = False
                    log.info(f"[GOES] Event detected: {event['class']} at {event['onset_time']}")

        return events


# ---------------------------------------------------------------------------
# Correlation Engine
# ---------------------------------------------------------------------------

class AnomalyCorrelator:
    """
    Cross-correlates GOES flare events with our ensemble detector's alert log
    to compute performance metrics (true positives, false positives, etc.)

    Reads from:
      logs/anomaly_log.jsonl  — append-only log of all anomaly_status.json states
    Writes to:
      reports/goes_correlation.json  — used by ai_reporter.py
    """

    def __init__(
        self,
        anomaly_log_path:    str = DEFAULT_ANOMALY_LOG_PATH,
        correlation_path:    str = DEFAULT_CORR_PATH,
        match_window_s:      int = 300,
    ):
        self.log_path   = Path(anomaly_log_path)
        self.corr_path  = Path(correlation_path)
        self.window     = match_window_s

    def _load_anomaly_log(self) -> List[Dict]:
        if not self.log_path.exists():
            return []
        entries = []
        with open(self.log_path) as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
        return entries

    def _parse_ts(self, iso_str: str) -> float:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp()

    def correlate(self, goes_events: List[Dict]) -> Dict:
        """
        Matches GOES flare events to our detector's alerts within ±match_window_s.

        Returns performance summary dict.
        """
        alerts   = [e for e in self._load_anomaly_log() if e.get("is_anomaly")]
        alert_ts = [self._parse_ts(a["timestamp"]) for a in alerts]
        goes_ts  = [self._parse_ts(e["onset_time"]) for e in goes_events]

        tp = 0  # True positives: GOES event + our alert within window
        for gt in goes_ts:
            matched = any(abs(at - gt) <= self.window for at in alert_ts)
            if matched: tp += 1

        fp = max(0, len(alerts) - tp)  # Our alerts with no corresponding GOES event
        fn = max(0, len(goes_ts) - tp) # GOES events we missed

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-6)

        result = {
            "goes_events_detected": len(goes_events),
            "our_alerts_fired":     len(alerts),
            "true_positives":       tp,
            "false_positives":      fp,
            "false_negatives":      fn,
            "precision":            round(precision, 4),
            "recall":               round(recall, 4),
            "f1_score":             round(f1, 4),
            "goes_events":          goes_events,
            "generated_at":         datetime.now(timezone.utc).isoformat(),
        }

        self.corr_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.corr_path, "w") as f:
            json.dump(result, f, indent=2)

        log.info(
            f"[Correlator] TP={tp} FP={fp} FN={fn} | "
            f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}"
        )
        return result


# ---------------------------------------------------------------------------
# Main Daemon
# ---------------------------------------------------------------------------

class GOESCorrelatorDaemon:
    """
    Long-running daemon that:
      1. Polls NOAA GOES every poll_interval_s seconds
      2. Detects flare events in the time series
      3. Writes goes_status.json (consumed by mission_control.py)
      4. Writes correlation report (consumed by ai_reporter.py)

    Run with: python goes_correlator.py
    Or as a systemd service / nohup process alongside other daemons.
    """

    def __init__(
        self,
        poll_interval_s:  int = 60,
        status_path:      str = DEFAULT_STATUS_PATH,
        anomaly_log_path: str = DEFAULT_ANOMALY_LOG_PATH,
    ):
        self.interval      = poll_interval_s
        self.status_path   = Path(status_path)
        self.fetcher       = GOESFetcher()
        self.parser        = GOESParser()
        self.detector      = GOESFlareDetector(min_class="B")
        self.correlator    = AnomalyCorrelator(anomaly_log_path)

    def _write_status(self, events: List[Dict], latest_flux: float) -> None:
        """Updates goes_status.json for mission_control.py's SFTP reader."""
        status = {
            "latest_xrsb_flux":    latest_flux,
            "latest_class":        GOESParser.subclass(latest_flux),
            "active_flare":        GOESParser.classify(latest_flux) in ["M", "X"],
            "recent_events_6h":    events[-5:],   # Last 5 events
            "event_count_6h":      len(events),
            "last_updated":        datetime.now(timezone.utc).isoformat(),
        }
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_path, "w") as f:
            json.dump(status, f, indent=2)

    def run_once(self) -> None:
        data = self.fetcher.fetch()
        if data is None:
            log.warning("[GOES] No data available.")
            return

        timestamps, flux = self.parser.parse(data, channel="B")
        if len(flux) == 0:
            log.warning("[GOES] Parsed flux array is empty.")
            return

        events = self.detector.detect(timestamps, flux)
        self.correlator.correlate(events)
        self._write_status(events, float(flux[-1]))

        log.info(
            f"[GOES] Latest XRS-B: {GOESParser.subclass(flux[-1])} "
            f"({flux[-1]:.2e} W/m²) | "
            f"{len(events)} events in last 6h"
        )

    def run(self) -> None:
        log.info(f"[GOES] Daemon started — polling every {self.interval}s")
        while True:
            try:
                self.run_once()
            except Exception as e:
                log.error(f"[GOES] Daemon error: {e}", exc_info=True)
            time.sleep(self.interval)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--once",     action="store_true", help="Run once and exit")
    p.add_argument("--interval", type=int, default=60)
    args = p.parse_args()

    daemon = GOESCorrelatorDaemon(poll_interval_s=args.interval)

    if args.once:
        daemon.run_once()
    else:
        daemon.run()
