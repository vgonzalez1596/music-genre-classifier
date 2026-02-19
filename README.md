# Project #3: Music Genre Classifier Using Machine Learning.

This repository contains a machine learning project that predicts a song’s genre from audio features using the Spotify Tracks dataset (Kaggle).

## Project Goal
Determine whether quantitative song metrics (e.g., tempo, loudness, danceability) can accurately classify genre.

---

## Approach
- Filtered dataset to 5 distinct genres: acoustic, country, hip-hop, metal, tango
- Feature scaling + train/test split (80/20)
- Trained and evaluated a Random Forest classifier
- Saved evaluation outputs (metrics + plots) for reproducibility

---

## Results
- Random Forest accuracy: **~0.84** on the 5-genre subset  
- Outputs generated in `results/`:
  - `metrics.txt`
  - confusion matrix
  - feature importance plot

---

## Example Outputs
[Confusion Matrix](results/confusion_matrix.png)

[Feature Importance](results/feature_importance.png)

---

## Tools & Packages
Python, pandas, scikit-learn, matplotlib

---

## Repository Structure
`data/` data organization notes

`results/` processed outputs and intermediate objects

`genre_classifier.py` analysis script

---

## Data availability
See `data/` for instructions to download the Spotify tracks list from Kaggle.
