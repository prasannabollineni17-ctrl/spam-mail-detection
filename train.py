import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def find_dataset():
    candidates = [
        "spam_dataset.csv",
        "mail_data.csv",
        "spam.csv",
        os.path.join("data", "spam_dataset.csv"),
        os.path.join("data", "mail_data.csv"),
        os.path.join("data", "spam.csv")
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    # Fallback to any .csv in current directory or data/
    for file in os.listdir("."):
        if file.endswith(".csv"):
            return file
    return None

def load_and_preprocess_data(file_path):
    print(f"[*] Loading dataset from: {file_path}")
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1")

    print(f"[*] Raw dataset shape: {df.shape}")
    print(f"[*] Columns detected: {list(df.columns)}")

    # Detect text column
    text_col = None
    for col in ["message_content", "Message", "message", "text", "v2", "email", "content", "SMS"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        text_col = df.columns[0]

    # Detect label column
    label_col = None
    for col in ["is_spam", "Category", "category", "label", "v1", "target", "class"]:
        if col in df.columns and col != text_col:
            label_col = col
            break
    if label_col is None:
        label_col = df.columns[1]

    print(f"[*] Using '{text_col}' as Text column and '{label_col}' as Label column")

    # Clean null values
    df = df.dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].astype(str)

    # Standardize labels to 0 (Ham) and 1 (Spam)
    unique_labels = df[label_col].unique()
    print(f"[*] Detected label classes: {unique_labels}")

    def standardize_label(val):
        str_val = str(val).strip().lower()
        if str_val in ["1", "spam", "true", "yes"]:
            return 1
        return 0

    df["label_num"] = df[label_col].apply(standardize_label)

    spam_count = (df["label_num"] == 1).sum()
    ham_count = (df["label_num"] == 0).sum()
    print(f"[*] Distribution: {ham_count} Ham (Legitimate) / {spam_count} Spam ({spam_count / len(df) * 100:.1f}% spam)")

    return df[text_col], df["label_num"]

def train_model():
    dataset_path = find_dataset()
    if not dataset_path:
        print("[!] Error: No dataset CSV file found in project directory.")
        print("[!] Please place 'spam_dataset.csv' in the project folder and run again.")
        sys.exit(1)

    X_text, y = load_and_preprocess_data(dataset_path)

    # 80/20 train/test split with stratification
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[*] Training samples: {len(X_train_raw)}, Testing samples: {len(X_test_raw)}")

    # Feature extraction with TF-IDF
    print("[*] Extracting features using TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_features=5000,
        ngram_range=(1, 2)
    )
    X_train_features = vectorizer.fit_transform(X_train_raw)
    X_test_features = vectorizer.transform(X_test_raw)
    print(f"[*] Feature matrix shape: {X_train_features.shape}")

    # Model training: Support Vector Machine (Linear SVM with probability calibration)
    print("[*] Training Support Vector Machine (Linear SVM with probability calibration)...")
    base_svm = SVC(kernel="linear", random_state=42, C=1.0)
    svm_model = CalibratedClassifierCV(base_svm, ensemble=False)
    svm_model.fit(X_train_features, y_train)

    # Evaluation
    print("\n" + "="*50)
    print("           MODEL EVALUATION RESULTS           ")
    print("="*50)

    train_preds = svm_model.predict(X_train_features)
    test_preds = svm_model.predict(X_test_features)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"[*] Training Accuracy : {train_acc * 100:.2f}%")
    print(f"[*] Testing Accuracy  : {test_acc * 100:.2f}%")
    print("\n[*] Detailed Classification Report (Test Set):")
    print(classification_report(y_test, test_preds, target_names=["Ham (Legitimate)", "Spam"]))

    print("[*] Confusion Matrix:")
    cm = confusion_matrix(y_test, test_preds)
    print(f"    True Negative (Ham correctly caught) : {cm[0][0]}")
    print(f"    False Positive (Ham marked as Spam) : {cm[0][1]}")
    print(f"    False Negative (Spam missed)        : {cm[1][0]}")
    print(f"    True Positive (Spam correctly caught): {cm[1][1]}")

    # Save artifacts
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "spam_svm_model.pkl")
    vectorizer_path = os.path.join("model", "tfidf_vectorizer.pkl")

    joblib.dump(svm_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"\n[+] Saved trained model to: {model_path}")
    print(f"[+] Saved vectorizer to   : {vectorizer_path}")
    print("[+] Training completed successfully!")

if __name__ == "__main__":
    train_model()
