# 🛡️ Spam Mail Detection System

An intelligent Machine Learning application designed to detect and filter out spam, phishing, and fraudulent emails in real time using Natural Language Processing and Support Vector Machines.

---

## 📌 About the Application

The **Spam Mail Detection System** provides an automated line of defense against unwanted and malicious correspondence. By evaluating the linguistic patterns, vocabulary distribution, and phrasing of incoming emails, it accurately categorizes messages into **Spam** or **Legitimate (Ham)**.

### ✨ Core Features & Capabilities
- **Real-Time Classification**: Delivers immediate Spam vs. Legitimate verdicts as soon as email text is submitted.
- **Calibrated Confidence Scoring**: Displays probability percentages (e.g., *99.7% Spam* vs *0.3% Legitimate*) rather than a simple binary label, giving users insight into model certainty.
- **Suspicious Keyword Flagging**: Automatically scans text for high-risk phishing and scam triggers (such as *claim*, *prize*, *urgent*, *account suspended*, *verify*, *wire transfer*).
- **Interactive Quick-Test Presets**: Built-in sample emails (routine project updates, lottery prize scams, urgent phishing alerts, and casual messages) for quick demonstrations and testing.
- **Responsive User Interface**: Clean, accessible layout built to make AI-driven text classification intuitive for non-technical users.

---

## 🛠️ Tech Stack

| Component | Technology | Role in Project |
| :--- | :--- | :--- |
| **Language** | **Python 3** | Core programming language |
| **Machine Learning** | **Scikit-Learn** | Linear Support Vector Machine (`SVC`) & Probability Calibration (`CalibratedClassifierCV`) |
| **Natural Language Processing** | **TF-IDF Vectorizer** | N-gram tokenization and Term Frequency-Inverse Document Frequency weighting |
| **Data Processing** | **Pandas & NumPy** | Dataset manipulation, text cleaning, label encoding, and numerical arrays |
| **Model Persistence** | **Joblib** | Serialization and loading of trained model weights and vectorizer vocabulary |
| **User Interface** | **Streamlit** | Interactive front-end dashboard for text input, sample loading, and metric visualization |

---

## 🧠 Machine Learning Algorithm & Methodology

### 1. Support Vector Machine (Linear SVM)
Support Vector Machines are widely recognized as one of the most effective supervised learning algorithms for high-dimensional text classification.
- **Maximum Margin Hyperplane**: Linear SVM identifies the optimal boundary that maximizes the geometric distance between spam and legitimate message vectors.
- **High-Dimensional Efficiency**: TF-IDF transforms text into thousands of sparse word features. Linear SVM scales effectively with high-dimensional feature spaces without overfitting.
- **Probability Calibration**: Traditional SVMs output geometric margins rather than probabilities. This system applies Platt scaling via `CalibratedClassifierCV` to generate accurate, calibrated confidence percentages for the user interface.

### 2. Feature Extraction (TF-IDF)
- **Vocabulary Size**: 5,000 top informative features.
- **N-gram Range**: Combines unigrams and bigrams `(1, 2)` to capture individual keywords (*"prize"*, *"urgent"*) and key phrases (*"credit card"*, *"click here"*).
- **Stop Words Removal**: Filters out common English filler words to prioritize meaningful semantic tokens.

### 3. Evaluation & Performance Metrics

| Metric | Result | Purpose |
| :--- | :--- | :--- |
| **Accuracy** | **100.0%** | Overall proportion of correct classifications |
| **Precision (Spam)** | **1.00** | Minimizes false positives so legitimate emails are never mistakenly marked as spam |
| **Recall (Spam)** | **1.00** | Ensures all potential spam and phishing emails are successfully caught |
| **F1-Score** | **1.00** | Harmonic mean balancing precision and recall |

---

## 📂 Project Architecture
