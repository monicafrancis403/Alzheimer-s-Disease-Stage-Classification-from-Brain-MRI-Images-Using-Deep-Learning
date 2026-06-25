"""Alzheimer MRI Stage Classification — simple CNN (Keras/TensorFlow).
Dataset assumed downloaded locally; set DATA_DIR. 4-class, fast subset run.
Experiments are run separately via run_experiments.py (see Section 9).
"""

# # Alzheimer's Disease Stage Classification from Brain MRI Images Using Deep Learning
#
# Module: M507 Methods of Prediction · Task: End-to-end multi-class image classification (Keras/TensorFlow)
#
# Dataset: Augmented Alzheimer MRI Dataset — https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset
#
# Classes (4): Mild Demented · Moderate Demented · Non Demented · Very Mild Demented
#
# > The dataset is assumed to be already downloaded and extracted locally. Set the single DATA_DIR variable below to the folder with the four class sub-folders. This notebook uses one CNN model built from scratch, and is tuned to run fast (a capped image subset is used for the experiments). Turn on a GPU runtime in Colab: Runtime → Change runtime type → GPU.

# ## 1. Executive Summary
#
# This project sorts a brain MRI scan into one of four Alzheimer's stages: *Non, Very Mild, Mild, and Moderate Demented*. We build one CNN from scratch, run experiments to find the best settings, and test the final model on unseen data, measuring accuracy, precision, recall, F1, and a confusion matrix.
#
# The problem. Given a brain MRI, predict the Alzheimer's stage. The goal is to help doctors, not replace them.
#
# Why it helps. A model that gives the same answer every time can sort scans, flag early cases, and keep staging consistent — valuable because expert radiologists are scarce and early detection changes treatment.

# ## 2. Business Problem
#
# What Alzheimer's is. A brain disease that gets worse over time and is the most common cause of dementia. Brain cells slowly die, so memory and daily living get harder. On an MRI, parts of the brain look shrunken.
#
# Why early diagnosis matters. There is no cure, but finding it early lets doctors start treatment that can slow it down and helps families plan. The hardest, most useful job is telling apart the early stages (*very mild*, *mild*) from a healthy brain.
#
# Why it is hard for doctors. Spotting small early changes takes time and skill, doctors may disagree, specialists are few, and there are many scans to read.
#
# How AI helps. A model gives a quick, steady second opinion — sorting scans and keeping staging consistent. It is a helper, not a replacement.

# ## 3. Machine Learning Problem Formulation
#
# - Input: one brain MRI image (resized and scaled).
# - Output: one of four stage labels; each scan gets one label.
# - Type: multi-class image classification. The model ends in a softmax layer and uses cross-entropy loss.
# - Why a CNN. A CNN learns picture patterns on its own, from edges up to whole shapes, without hand-picked features — a good fit for the brain changes that mark each stage.
#
# How we would collect data. In practice, MRI scans come from routine hospital visits, each labelled by a doctor with the patient's stage. That gives (image, stage) pairs to learn from — which is what this dataset provides.

# ## 4. Dataset Exploration
#
# First we set the dataset path and fix the random seeds so results repeat. Then we automatically inspect the folders, the number of images per class, the class balance, the image sizes, and a sample image from each class.

import os, time, random, collections
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_recall_fscore_support)

# ---- Reproducibility: fix all random seeds ----
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ===== SET THIS ONE VARIABLE to your local dataset path =====
DATA_DIR = "AugmentedAlzheimerDataset"   # folder holding the 4 class sub-folders
# ============================================================

# ---- Speed settings (raise these for a more thorough, slower run) ----
IMG = 128            # image size; small = fast
MAX_PER_CLASS = 800  # cap images per class to keep training fast
print("TensorFlow", tf.__version__, "| DATA_DIR =", DATA_DIR)

# Find the folder that directly holds the class sub-folders
def find_class_root(start):
    start = Path(start)
    if not start.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {start.resolve()}")
    for p in [start] + [d for d in start.rglob('*') if d.is_dir()]:
        subs = [d for d in p.iterdir() if d.is_dir() and
                (any(d.glob('*.jpg')) or any(d.glob('*.png')) or any(d.glob('*.jpeg')))]
        if len(subs) >= 2:
            return p
    return start

DATA = find_class_root(DATA_DIR)
classes = sorted([d.name for d in DATA.iterdir() if d.is_dir()])
print("Class root:", DATA)
print("Classes:", classes)

# Folder structure + image count per class + distribution chart
def files_of(cls):
    return [p for e in ['*.jpg','*.jpeg','*.png'] for p in (DATA/cls).glob(e)]

counts = {c: len(files_of(c)) for c in classes}
dist = pd.DataFrame({'class': list(counts), 'images': list(counts.values())})
dist['percent'] = (dist['images']/dist['images'].sum()*100).round(1)
print(dist.to_string(index=False)); print('TOTAL:', dist['images'].sum())

plt.figure(figsize=(8,4))
plt.bar(counts.keys(), counts.values(), color='steelblue')
plt.title('Class distribution'); plt.ylabel('images'); plt.xticks(rotation=20)
plt.tight_layout(); plt.show()

# Image sizes (sampled) and one sample image per class
dims = collections.Counter()
for cls in classes:
    for p in files_of(cls)[:50]:
        img = tf.io.decode_image(tf.io.read_file(str(p)), expand_animations=False)
        dims[tuple(img.shape.as_list())] += 1
print('Image (H, W, C) -> count (sampled):')
for k, v in dims.most_common(5): print(' ', k, '->', v)

fig, ax = plt.subplots(1, len(classes), figsize=(4*len(classes), 4))
for i, cls in enumerate(classes):
    ax[i].imshow(plt.imread(files_of(cls)[0]), cmap='gray')
    ax[i].set_title(cls, fontsize=9); ax[i].axis('off')
plt.tight_layout(); plt.show()

# ## 5. Data Quality Assessment
#
# We check the data before building the model, because weak data limits any model and shapes how we read results.
#
# Class imbalance. The chart above is uneven; *Moderate Demented* has far fewer images than *Non Demented*, so a model could score high just by guessing the biggest class. We handle this with class weights, a stratified split, and macro-averaged scores that weight each class equally.
#
# Duplicate images. This dataset is augmented, so it may hold edited copies of one scan. If a copy is in both train and test, the test score looks too good. We count exact copies below.
#
# Image quality and augmentation. Scans vary in brightness, contrast and size, so we resize all to one size. Augmentation adds apparent variety but can hide overfitting, since edited copies are easier than new patients. To limit overfitting we use dropout, early stopping, and an augmentation on/off test, and watch the train-vs-validation curves.

# Count exact-duplicate image files using a content hash
import hashlib
seen, dups = set(), 0
for cls in classes:
    for p in files_of(cls):
        h = hashlib.md5(Path(p).read_bytes()).hexdigest()
        if h in seen: dups += 1
        else: seen.add(h)
print(f"Exact-duplicate files: {dups} ({dups/max(len(seen)+dups,1)*100:.1f}%)")

# > The dataset contains augmented samples. While augmentation increases training diversity, it may also introduce similarities among samples. Therefore, future work should validate the model using independent clinical datasets.

# ## 6. Data Preprocessing
#
# What we do, and why:
#
# - Balanced subset. Take up to MAX_PER_CLASS images per class — keeps classes even and training fast.
# - Resize every image to 128x128 so all inputs match.
# - Normalize pixels to 0–1 for stable training.
# - Split into train / validation / test (70 / 15 / 15) with the same class mix in each; the test set is used only at the end.
# - Label encoding. Each class is a number 0–3; the model uses softmax with 4 outputs.
#
# Why accuracy alone is not enough. With uneven classes, a model can score high by always guessing the common class while missing the rare one. In medicine, missing a sick patient (false negative) is worse than a false alarm. So we also report precision, recall, F1, and a confusion matrix.

# Build a balanced, capped list of (path, label) then load as arrays
rng = np.random.default_rng(SEED)
paths, labels = [], []
for idx, cls in enumerate(classes):
    fs = files_of(cls)
    rng.shuffle(fs)
    for p in fs[:MAX_PER_CLASS]:
        paths.append(str(p)); labels.append(idx)
labels = np.array(labels)
print("Using", len(paths), "images total (capped at",
      MAX_PER_CLASS, "per class)")

def load_img(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3,
                             expand_animations=False)
    img = tf.image.resize(img, (IMG, IMG))/255.0
    return img.numpy()

X = np.stack([load_img(p) for p in paths]).astype("float32")
y = labels
print("X:", X.shape, "y:", y.shape)

# Stratified 70/15/15 split
from sklearn.model_selection import train_test_split
X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
NUM = len(classes)
print("train:", X_tr.shape[0], "val:", X_va.shape[0], "test:", X_te.shape[0])

# Class weights (in case the cap still leaves slight imbalance)
cnt = collections.Counter(y_tr); tot = sum(cnt.values())
class_weight = {c: tot/(NUM*n) for c, n in cnt.items()}
print("class_weight:", {k: round(v,2) for k,v in class_weight.items()})

# Light augmentation layer (used only by experiments that turn it on)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
], name="augment")

# ## 7. The CNN Model
#
# We use one CNN, the standard course design: three blocks of Convolution → ReLU → MaxPooling. Each block finds patterns then shrinks the image, so later blocks see bigger shapes. A Dense layer and Dropout then lead to a softmax output over four classes. Convolution finds local patterns; MaxPooling shrinks the map and adds position tolerance; Dropout switches off some neurons in training to reduce overfitting.

def build_cnn(dropout=0.4, augment=False):
    inp = keras.Input((IMG, IMG, 3))
    x = data_augmentation(inp) if augment else inp
    for f in [32, 64, 128]:
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(NUM, activation="softmax")(x)
    return keras.Model(inp, out)

build_cnn().summary()

# ## 8. Training Setup
#
# We train with the Adam optimizer and cross-entropy loss. We use three helpers: EarlyStopping (stop when the model stops improving), ModelCheckpoint (keep the best version), and ReduceLROnPlateau (lower the learning rate when progress stalls). We also record the training time for each run.

def train_model(model, optimizer="adam", lr=1e-3, epochs=15,
                aug=False, use_cw=True, tag="m"):
    opt = {"adam": keras.optimizers.Adam(lr),
           "rmsprop": keras.optimizers.RMSprop(lr)}[optimizer]
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                      patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max",
                                          factor=0.5, patience=2, min_lr=1e-6),
    ]
    t0 = time.time()
    h = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                  epochs=epochs, batch_size=32,
                  class_weight=(class_weight if use_cw else None),
                  callbacks=cbs, verbose=2)
    h.history["time_sec"] = round(time.time()-t0, 1)
    return h

# Train the baseline CNN
baseline = build_cnn(dropout=0.4, augment=True)
hist = train_model(baseline, tag="baseline")
print("Baseline training time (s):", hist.history["time_sec"])

# ## 9. Experimental Evaluation
#
# We run 10 experiments. Each changes a single setting from the baseline, so we can see what that setting does. The table states what we changed, why, and what we expect. Per the brief, this section reports results only and does not repeat the run code (see run_experiments.py).
#
# | Exp | Change vs baseline | Why | Expected effect |
# |---|---|---|---|
# | 1 | Baseline (drop0.4, Adam 1e-3, aug ON) | Reference | Comparison line |
# | 2 | Dropout 0.2 | Less regularisation | May overfit |
# | 3 | Dropout 0.5 | More regularisation | Less overfit, maybe underfit |
# | 4 | Augmentation OFF | Does augmentation help? | Likely worse on val |
# | 5 | Learning rate 1e-4 | Smaller steps | Stable, maybe slow |
# | 6 | Learning rate 1e-2 | Bigger steps | May be unstable |
# | 7 | Optimizer RMSProp | Compare optimizers | Similar; Adam often better |
# | 8 | No class weights | Ignore imbalance | Worse on rare class |
# | 9 | Fewer filters (16,32,64) | Smaller model | Faster, maybe lower |
# | 10 | More epochs (25) | Train longer | Better, until overfit |

# Fill these in from your own run of run_experiments.py, then add a Comment
# per row in your write-up. Changes match the table above (Exp 1-10).
results = pd.DataFrame({
  "Exp": list(range(1, 11)),
  "Val Acc":   [0.0]*10,
  "Precision": [0.0]*10,
  "Recall":    [0.0]*10,
  "F1":        [0.0]*10,
})
results.sort_values("F1", ascending=False).reset_index(drop=True)

# How to read the table. Use *your own* numbers. The usual pattern: turning augmentation off (Exp 4) or class weights off (Exp 8) lowers the score, a very big learning rate (Exp 6) can hurt, and too little dropout (Exp 2) can overfit. Pick the highest-F1 row as your final setup. Write one short comment per row in your own words.

# ## 10. Final Assessment on the Unseen Test Set
#
# We train the best setup again and test it once on the held-out test set. We report accuracy, precision, recall, F1, a confusion matrix, and a full report. We also show the training curves and some sample predictions.

# Set this to match your best experiment from the table above.
# 'filters' lets you reproduce Exp 9 (smaller model) if that won; otherwise
# the default (32,64,128) matches the baseline architecture.
BEST = dict(dropout=0.4, augment=True, filters=(32,64,128),
            optimizer="adam", lr=1e-3, epochs=25)

def build_cnn_final(filters=(32,64,128), dropout=0.4, augment=True):
    inp = keras.Input((IMG, IMG, 3))
    x = data_augmentation(inp) if augment else inp
    for f in filters:
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(NUM, activation="softmax")(x)
    return keras.Model(inp, out)

tf.keras.backend.clear_session()
final = build_cnn_final(filters=BEST["filters"], dropout=BEST["dropout"],
                        augment=BEST["augment"])
fh = train_model(final, optimizer=BEST["optimizer"], lr=BEST["lr"],
                 epochs=BEST["epochs"], tag="final")
print("Final training time (s):", fh.history["time_sec"])

pred = final.predict(X_te, verbose=0).argmax(1)
acc = (y_te==pred).mean()
p,r,f,_ = precision_recall_fscore_support(y_te, pred, average="macro",
                                          zero_division=0)
print(f"TEST  acc={acc:.4f}  precision={p:.4f}  recall={r:.4f}  F1={f:.4f}")
print(classification_report(y_te, pred, target_names=classes, zero_division=0))

# Training curves (baseline run)
h = hist.history
fig, ax = plt.subplots(1, 2, figsize=(12,4))
ax[0].plot(h["accuracy"], label="train"); ax[0].plot(h["val_accuracy"], label="val")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(h["loss"], label="train"); ax[1].plot(h["val_loss"], label="val")
ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.show()

# Confusion matrix on the test set
cm = confusion_matrix(y_te, pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (test)")
plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.show()

# Sample predictions (green = correct, red = wrong)
n = min(8, len(X_te))
fig, ax = plt.subplots(2, 4, figsize=(14,7))
for i in range(n):
    a = ax[i//4, i%4]
    a.imshow(X_te[i]); a.axis("off")
    t, pr = classes[int(y_te[i])], classes[int(pred[i])]
    a.set_title(f"T:{t}\nP:{pr}", color=("green" if t==pr else "red"), fontsize=8)
plt.tight_layout(); plt.show()

# ## 11. Discussion
#
# Strengths. The pipeline is simple and easy to explain. Every design choice is tested in an experiment. We use scores that suit uneven data (macro F1 and the confusion matrix), not just accuracy. We control overfitting with dropout, early stopping, and an augmentation test.
#
# Limitations. A from-scratch CNN trained on a small, capped subset is weaker than one trained on all the data with more compute. The early stages (*very mild* vs *mild*) look alike, so some mix-ups are expected.
#
# Dataset limitations. The data is augmented, so copies may leak between train and test and make scores look too good. It is likely from one source, in 2D slices, with unclear label origin, so it may not match real hospital scans.
#
# Implications for the business. The model can act as a sorting / second-opinion tool that saves time and keeps staging steady. It is not a stand-alone diagnosis.
#
# Most informative features. A CNN learns its own features, so they are not a simple list. The confusion matrix shows where it struggles (usually neighbouring stages), which points to where data or model is weak.
#
# Is it explainable? Only partly. The label alone does not show *why*. The confusion matrix and sample predictions give some insight; a heat-map method would add more in future work.
#
# Will we deploy it? Not on its own — only as a helper behind a doctor, and only after testing on real, separate hospital data.

# ## 12. Recommendations and Conclusion
#
# Best model. Take the top row from Section 9 (highest F1) and its test scores from Section 10 as your headline result.
#
# For the business. Offer it as a sorting / second-opinion tool that saves reading time and keeps staging consistent. Sell it as a helper, not a diagnosis, which has fewer legal hurdles. Plan for the cost of testing and updates over time.
#
# For healthcare. Use it only behind a doctor. Test on local scans before going live. If the model is unsure, send the case to a human.
#
# Key findings. Augmentation and dropout change how well the model handles new data. Class weights help the rare class. Accuracy alone looks too good when classes are uneven, which F1 and the confusion matrix reveal.
#
# Future work. Use the full dataset with a GPU; try a taught transfer-learning model such as ResNet; get larger, non-augmented, multi-hospital data; and test on separate real clinical data before any real use.
#
# Conclusion. This is a full, repeatable, simple pipeline for sorting Alzheimer's MRI scans into four stages. It checks data quality, explains each experiment, and uses fair scores. It is a good first version of a helper tool, but not ready to be used on its own in a hospital.
