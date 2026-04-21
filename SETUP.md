# MATH 2320 Final Project — Setup Guide

**Team:** Deja Dunlap, Jenna Chow, Aditya Das, Undra Pillows

---

## Project Structure

```
MATH 2320 Final Project/
├── requirements.txt       ← Python packages to install
├── pca_analysis.py        ← Part 1: PCA
├── dataset.csv            ← you download this (see Step 2)
└── SETUP.md               ← this file
```

---

## Step 1 — Install Python & VS Code (if not already)

- Python: https://www.python.org/downloads/ (3.9 or later)
- VS Code: https://code.visualstudio.com/
- In VS Code, install the **Python extension** (search "Python" in the Extensions sidebar)

---

## Step 2 — Download the Dataset

1. Go to: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
2. Click **Download** (you'll need a free Kaggle account)
3. Unzip the downloaded file
4. Rename the CSV to `dataset.csv`
5. Place `dataset.csv` in the same folder as `pca_analysis.py`

---

## Step 3 — Open the project in VS Code

1. Open VS Code
2. Go to **File → Open Folder** and select the `MATH 2320 Final Project` folder
3. Open a terminal inside VS Code: **Terminal → New Terminal**

---

## Step 4 — Install packages

In the VS Code terminal, run:

```bash
pip install -r requirements.txt
```

This installs: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

---

## Step 5 — Run the PCA script

In the terminal:

```bash
python pca_analysis.py
```

This will:
- Load and clean the dataset
- Standardize the audio features
- Run PCA and print variance explained per component
- Save 4 plots to your folder:
  - `scree_plot.png` — how many PCs matter
  - `pca_loadings.png` — which features drive each PC
  - `pca_scatter.png` — songs in 2D PC space, colored by genre
  - `pca_biplot.png` — feature directions overlaid on PC scatter

---

## Notes

- The `GENRES_TO_USE` list in the script filters to 10 genres for cleaner plots.
  You can add/remove genres or comment out that block to use all genres.
- The scatter plot samples 3000 songs for readability; change `SAMPLE_SIZE` as needed.
- Next step will be diffusion maps (`diffusion_maps.py`) — coming soon!
