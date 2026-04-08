# Aditya-L1 Algorithm Upgrades — Integration Guide

## Step 1 — Train the VAE
`python train_vae.py --catalog ../aditya_l1_catalog.csv --image_dir ../processed_images --epochs 60`

## Step 2 — Optical Flow Watchdog
Flow tracking creates velocity fields parallel to the PNG frames using `FarnebackFlowExtractor`.

## Step 3 — GOES Correlation
Run `nohup python3 goes_correlator.py &` to sync with NOAA ground truth.
