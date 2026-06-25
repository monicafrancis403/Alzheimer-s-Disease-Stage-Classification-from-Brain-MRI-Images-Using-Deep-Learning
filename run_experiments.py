"""run_experiments.py — runs the 10 pipeline experiments for the Alzheimer's
MRI notebook and prints a results table (validation accuracy, precision, recall,
F1, training time).

The brief asks the notebook to *report* the experiment results without repeating
the run code. So the experiment loop lives here, separate from the notebook.
Run this once, then copy the printed numbers into the results table in
Section 9 of the notebook.

Usage:
    python run_experiments.py
Set DATA_DIR, IMG and MAX_PER_CLASS below to match the notebook.
"""
import os, time, random, collections
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

# ---- reproducibility ----
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ---- settings (match the notebook) ----
DATA_DIR = "AugmentedAlzheimerDataset"
IMG = 128
MAX_PER_CLASS = 800

# ---- locate class folders ----
from pathlib import Path
def find_class_root(start):
    start = Path(start)
    for p in [start] + [d for d in start.rglob('*') if d.is_dir()]:
        subs = [d for d in p.iterdir() if d.is_dir() and
                (any(d.glob('*.jpg')) or any(d.glob('*.png')) or any(d.glob('*.jpeg')))]
        if len(subs) >= 2:
            return p
    return start

DATA = find_class_root(DATA_DIR)
classes = sorted([d.name for d in DATA.iterdir() if d.is_dir()])
NUM = len(classes)

def files_of(cls):
    return [p for e in ['*.jpg','*.jpeg','*.png'] for p in (DATA/cls).glob(e)]

# ---- build balanced subset ----
rng = np.random.default_rng(SEED)
paths, labels = [], []
for idx, cls in enumerate(classes):
    fs = files_of(cls); rng.shuffle(fs)
    for p in fs[:MAX_PER_CLASS]:
        paths.append(str(p)); labels.append(idx)
labels = np.array(labels)

def load_img(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    return (tf.image.resize(img, (IMG, IMG))/255.0).numpy()

X = np.stack([load_img(p) for p in paths]).astype("float32")
y = labels

X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

cnt = collections.Counter(y_tr); tot = sum(cnt.values())
class_weight = {c: tot/(NUM*n) for c, n in cnt.items()}

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
], name="augment")

def cnn_variant(filters=(32,64,128), dropout=0.4, augment=False):
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

def train_model(model, optimizer="adam", lr=1e-3, epochs=12, use_cw=True):
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
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs,
              batch_size=32, class_weight=(class_weight if use_cw else None),
              callbacks=cbs, verbose=2)
    return round(time.time()-t0, 1)

def score(model):
    pred = model.predict(X_va, verbose=0).argmax(1)
    p,r,f,_ = precision_recall_fscore_support(y_va, pred, average="macro", zero_division=0)
    return (y_va==pred).mean(), p, r, f

# (id, builder-kwargs, train-kwargs)
runs = [
 (1, dict(dropout=0.4, augment=True),  dict()),
 (2, dict(dropout=0.2, augment=True),  dict()),
 (3, dict(dropout=0.5, augment=True),  dict()),
 (4, dict(dropout=0.4, augment=False), dict()),
 (5, dict(dropout=0.4, augment=True),  dict(lr=1e-4)),
 (6, dict(dropout=0.4, augment=True),  dict(lr=1e-2)),
 (7, dict(dropout=0.4, augment=True),  dict(optimizer="rmsprop")),
 (8, dict(dropout=0.4, augment=True),  dict(use_cw=False)),
 (9, dict(dropout=0.4, augment=True, filters=(16,32,64)), dict()),
 (10,dict(dropout=0.4, augment=True),  dict(epochs=25)),
]

rows = []
for eid, bkw, tkw in runs:
    tf.keras.backend.clear_session()
    m = cnn_variant(**bkw)
    t = train_model(m, epochs=tkw.pop("epochs", 12), **tkw)
    a,p,r,f = score(m)
    rows.append({"Exp":eid, "Val Acc":round(a,4), "Precision":round(p,4),
                 "Recall":round(r,4), "F1":round(f,4), "Time(s)":t})

results = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
print("\n=== Experiment results (copy these into Section 9 of the notebook) ===")
print(results.to_string(index=False))
print("\nBest experiment by F1:", int(results.iloc[0]["Exp"]))
