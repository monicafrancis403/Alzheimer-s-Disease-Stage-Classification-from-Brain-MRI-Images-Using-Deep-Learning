# Alzheimer's MRI Stage Classification (Simple CNN)

A simple, fast deep-learning pipeline that classifies brain MRI scans into four
Alzheimer's stages using one CNN in Keras/TensorFlow. Built for the Gisma M507
Methods of Prediction individual project.

Multi-class image classification (4 classes): Mild Demented, Moderate Demented,
Non Demented, Very Mild Demented. Primary metrics are macro precision / recall /
F1 and a confusion matrix, not accuracy alone.

## Design choices (matches the lectures and the brief)
- ONE model: a custom CNN from scratch (Conv/Pool/Dense), the architecture taught
  in the course. No MobileNet/EfficientNet/Grad-CAM (those are not in the lectures).
- Fast to run: uses a capped, balanced subset (MAX_PER_CLASS) at 128x128, so it
  finishes in ~20-30 min on a Colab GPU instead of several hours.
- Notebook kept under 20,000 characters (brief requirement). The experiment loop
  lives in run_experiments.py so Section 9 reports results only (brief requirement).

## Dataset (local)
Augmented Alzheimer MRI Dataset (uraninjo):
https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset
Download and extract it yourself, then set the single `DATA_DIR` variable near the
top of the notebook. The notebook does NOT call the Kaggle API.

## How to run
1. `pip install -r requirements.txt`
2. In Colab, set runtime to GPU: Runtime -> Change runtime type -> GPU (T4).
3. Set DATA_DIR in both the notebook and run_experiments.py.
4. Run `python run_experiments.py` once. Copy the printed numbers into the
   results table in Section 9 of the notebook, and write a short comment per row.
5. Run the notebook top to bottom. Set BEST (final cell) to your top experiment.
6. Export the notebook to HTML for submission (File -> Download -> .html).

## Files
- `alzheimer_mri_final.ipynb` — the submission notebook (under 20k chars).
- `run_experiments.py`        — runs the 10 experiments, prints the results table.
- `alzheimer_mri_final.py`    — script version of the notebook.

## Why the earlier version was slow (4-5 hours)
It trained 12 models, several of them large pretrained networks at 224x224 on the
full augmented dataset. This version uses one small CNN on a capped subset, and
ensures the GPU runtime is on.

## Note
This is coursework. The write-up must reflect your own understanding of results
from your own run. The dataset is augmented, so mention possible duplicate leakage
as a limitation (already noted in the notebook).
