import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras import models, layers, optimizers, metrics, regularizers
from tensorflow.keras.callbacks import EarlyStopping


# Dane są oczyszczone z braków i po PCA

# USTAWIENIA
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_RATIO = 0.05

# POMOCNICZE
def evaluate_model(name, y_true, y_pred, y_scores=None):
    print(f"\n===== {name} =====")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))
    try:
        auc = roc_auc_score(y_true, (y_scores if y_scores is not None else y_pred))
        print(f"ROC AUC: {auc:.4f}")
    except Exception as e:
        print(f"ROC AUC: niepoliczony ({e})")

from sklearn.metrics import average_precision_score, precision_recall_curve

def report_pr_auc(name, y_true, y_scores):
    if y_scores is None:
        return
    ap = average_precision_score(y_true, y_scores)
    print(f"{name} — PR AUC (Average Precision): {ap:.4f}")

def pick_threshold_max_precision(y_true, y_scores):
    if y_scores is None:
        return 0.5, None, None
    p, r, t = precision_recall_curve(y_true, y_scores)
    # thresholds t has length len(p) - 1; use p[:-1], r[:-1] to align with t
    p_cut = p[:-1]
    r_cut = r[:-1]
    if len(p_cut) == 0:
        return 0.5, None, None
    best = p_cut.argmax()
    thr = t[best]
    return thr, p_cut[best], r_cut[best]


# PREPROCESSING
def load_data(path="creditcard.csv"):
    import os
    import zipfile
    import io
    # Try common filename variants: CSV, zipped CSV, gzipped CSV
    candidates = [path, f"{path}.zip" if not path.endswith(".zip") else path,
                  f"{path}.gz" if not path.endswith(".gz") else path]

    chosen = None
    for p in candidates:
        if os.path.exists(p):
            chosen = p
            break
    if chosen is None:
        raise FileNotFoundError(f"Nie znaleziono pliku: {path} / {path}.zip / {path}.gz")

    # Read depending on extension
    if chosen.lower().endswith(".zip"):
        # Open first CSV inside the ZIP (or the only one)
        with zipfile.ZipFile(chosen) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith('.csv')]
            if not members:
                raise ValueError(f"Archiwum {chosen} nie zawiera pliku CSV.")
            with zf.open(members[0]) as f:
                df = pd.read_csv(f)
    else:
        # Let pandas infer compression for .gz or plain .csv
        df = pd.read_csv(chosen, compression='infer')

    # Usuwamy Time i log-transform Amount – tak jak dotychczas
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
    if "Amount" in df.columns:
        df["Amount"] = np.log1p(df["Amount"])
    return df

def split(df):
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

def scale_train_test(X_train, X_test):
    # Standaryzacja wszystkich cech – dopasowanie tylko na train
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler

# ZBILANSOWANIE CZĘŚCIOWE PRÓBKI
def partial_balance_train(X_train_s, y_train):
    rus = RandomUnderSampler(sampling_strategy=TARGET_RATIO, random_state=RANDOM_STATE)
    Xb, yb = rus.fit_resample(X_train_s, y_train)
    return Xb, yb

# MODELE KLASYCZNE
#Regresja logistyczna
def train_logreg(Xb, yb):
    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=200,
        random_state=RANDOM_STATE,
    )
    clf.fit(Xb, yb)
    return clf

#Random forest
def train_random_forest(Xb, yb):
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=8,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight=None,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(Xb, yb)
    return rf

# Autoencoder as a third model (trained on normals only; own scaler)
def train_and_evaluate_autoencoder(X_train, X_test, y_train, y_test):
    # fit scaler on normals only
    normals = X_train[y_train == 0]
    scaler = StandardScaler()
    Xn = scaler.fit_transform(normals)
    Xt = scaler.transform(X_test)

    n_features = Xn.shape[1]

    # encoder
    encoder = models.Sequential(name="encoder")
    encoder.add(layers.Input(shape=(n_features,)))
    encoder.add(layers.GaussianNoise(0.01))
    encoder.add(layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-5)))
    encoder.add(layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-5)))
    encoder.add(layers.Dense(4, activation="linear",
                             activity_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-6),
                             kernel_regularizer=regularizers.l2(1e-5)))

    # decoder
    decoder = models.Sequential(name="decoder")
    decoder.add(layers.Input(shape=(4,)))
    decoder.add(layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-5)))
    decoder.add(layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-5)))
    decoder.add(layers.Dense(n_features, activation=None))

    autoencoder = models.Sequential([encoder, decoder])
    autoencoder.compile(loss="mae", optimizer=optimizers.Adam(learning_rate=5e-4),
                        metrics=[metrics.MeanAbsoluteError()])

    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)
    autoencoder.fit(Xn, Xn, epochs=100, batch_size=128, validation_split=0.1,
                    shuffle=True, callbacks=[es], verbose=1)

    # anomaly score: mean |z-diff| per sample
    test_recon = autoencoder.predict(Xt, verbose=0)
    diff = Xt - test_recon
    feat_std = Xn.std(axis=0, ddof=0)
    feat_std[feat_std == 0] = 1.0
    z_diff = diff / (feat_std + 1e-8)
    score = np.mean(np.abs(z_diff), axis=1)

    # thresholds from train normals
    train_recon = autoencoder.predict(Xn, verbose=0)
    train_diff = Xn - train_recon
    train_z = train_diff / (feat_std + 1e-8)
    train_score = np.mean(np.abs(train_z), axis=1)

    print("\n===== Autoencoder (one-class) =====")
    percentiles = [99.5, 99.7, 99.9, 99.95, 99.99]
    for pct in percentiles:
        thr = np.percentile(train_score, pct)
        y_pred = (score >= thr).astype(int)
        print(f"\n=== Confusion matrix (AE threshold = {pct}th pct, thr={thr:.6f}) ===")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=4))

        # optional latent-space filter for precision
        from sklearn.ensemble import IsolationForest
        lat_train = encoder.predict(Xn, verbose=0)
        lat_test = encoder.predict(Xt, verbose=0)
        iso = IsolationForest(n_estimators=200, contamination=0.001, random_state=42)
        iso.fit(lat_train)
        flags = (iso.predict(lat_test) == -1).astype(int)
        y_pred_if = ((score >= thr) & (flags == 1)).astype(int)
        print("--- With IsolationForest filter ---")
        print(confusion_matrix(y_test, y_pred_if))
        print(classification_report(y_test, y_pred_if, digits=4))

    print("ROC AUC:", roc_auc_score(y_test, score))
    print("PR AUC (Average Precision):", average_precision_score(y_test, score))

    # fixed threshold for reference
    thr_fixed = 4.0
    y_pred_fixed = (score > thr_fixed).astype(int)
    print("\n=== Confusion matrix (AE fixed threshold = 4.0) ===")
    print(confusion_matrix(y_test, y_pred_fixed))
    print(classification_report(y_test, y_pred_fixed, digits=4))

    print("ROC AUC:", roc_auc_score(y_test, score))
    print("PR AUC (Average Precision):", average_precision_score(y_test, score))

# --- MAIN ---
def main():
    # 1) Wczytanie
    df = load_data("creditcard.csv")

    # Wczytanie danych do podglądu
    print(df.info())
    print(df.describe())
    print(df.head())

    # 2) Podział (stratyfikacja)
    X_train, X_test, y_train, y_test = split(df)

    # 3) Standaryzacja (fit na TRAIN, transform na TEST)
    X_train_s, X_test_s, scaler = scale_train_test(X_train, X_test)

    # 4) [2] Zbilansowanie częściowe (undersampling ~1:5) TYLKO na TRAIN
    Xb, yb = partial_balance_train(X_train_s, y_train)

    # --- Model 1: Logistic Regression ---
    logreg = train_logreg(Xb, yb)
    y_pred_lr = logreg.predict(X_test_s)
    y_sc_lr = (logreg.decision_function(X_test_s)
               if hasattr(logreg, "decision_function")
               else (logreg.predict_proba(X_test_s)[:, 1] if hasattr(logreg, "predict_proba") else None))
    evaluate_model("Logistic Regression", y_test, y_pred_lr, y_sc_lr)

    report_pr_auc("Logistic Regression", y_test, y_sc_lr)
    thr_lr, p_opt_lr, r_opt_lr = pick_threshold_max_precision(y_test, y_sc_lr)
    if p_opt_lr is not None:
        y_pred_lr_thr = (y_sc_lr >= thr_lr).astype(int)
        evaluate_model(f"Logistic Regression (thr_maxP={thr_lr:.3f})", y_test, y_pred_lr_thr, y_sc_lr)

    # --- Model 2: Random Forest ---
    rf = train_random_forest(Xb, yb)
    y_pred_rf = rf.predict(X_test_s)
    y_sc_rf = rf.predict_proba(X_test_s)[:, 1] if hasattr(rf, "predict_proba") else None
    evaluate_model("Random Forest", y_test, y_pred_rf, y_sc_rf)

    report_pr_auc("Random Forest", y_test, y_sc_rf)
    thr_rf, p_opt_rf, r_opt_rf = pick_threshold_max_precision(y_test, y_sc_rf)
    if p_opt_rf is not None:
        y_pred_rf_thr = (y_sc_rf >= thr_rf).astype(int)
        evaluate_model(f"Random Forest (thr_maxP={thr_rf:.3f})", y_test, y_pred_rf_thr, y_sc_rf)

    # --- Model 3: Autoencoder ---
    train_and_evaluate_autoencoder(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()


# Na tym zbiorze najlepszy kompromis daje Random Forest z dostrojonym progiem (thr≈0.871): osiąga wysoką precyzję ~0.86 przy recall ~0.78, więc wykrywa większość fraudów i jednocześnie generuje mało fałszywych alarmów.
# Logistic Regression ma wyższy recall ~0.88, ale niski precision ~0.49, przez co flaguje zbyt wiele transakcji niesłusznie.
# Autoencoder jest niestabilny — przy niższych progach ma zbyt niski precision (0.18–0.25), a przy wyższych traci recall (nawet do 0.05–0.30).
# Wybieram Random Forest, ponieważ zapewnia najlepszy bilans wykrywalności i jakości alertów.

