# 🛡️ Spam Mail Detection System using SVM & Streamlit

An end-to-end Machine Learning project to classify email and SMS messages as **Spam** or **Legitimate (Ham)** using a calibrated **Support Vector Machine (Linear SVM)** and **TF-IDF Vectorization**, complete with an interactive **Streamlit** web application.

---

## 📌 Project Overview

Spam emails can pose security risks such as phishing, malware delivery, and financial fraud. This project builds a machine learning pipeline that learns patterns and vocabulary from email text to accurately identify spam messages while ensuring legitimate business or personal emails are not mistakenly flagged (high precision).

### ✨ Key Features
- **Accurate Classification**: Powered by Support Vector Machine (`LinearSVC` / `SVC(kernel='linear')`) with calibrated probability scoring.
- **TF-IDF N-Gram Feature Extraction**: Captures single words and two-word phrases (unigrams & bigrams) with English stop-word filtering.
- **Interactive Web Interface**: Streamlit UI with quick-test sample emails, real-time confidence breakdowns, and suspicious keyword detection.
- **Production-Ready**: Exportable serialized artifacts (`.pkl`), robust error handling, and clean code structure ready for cloud deployment.

---

## 📂 Project Structure

```
Spam_Mail_Detection/
│
├── spam_dataset.csv          # Email dataset (message_content, is_spam)
├── train.py                  # Script to load data, train SVM, evaluate & save model
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules for virtual environments and caches
├── README.md                 # Project documentation & GitHub guide
│
├── model/                    # Generated after running train.py
│   ├── spam_svm_model.pkl    # Serialized trained SVM model
│   └── tfidf_vectorizer.pkl  # Serialized TF-IDF vectorizer
│
└── venv/                     # Python virtual environment (ignored by Git)
```

---

## ⚙️ Local Setup & Installation

### 1. Clone or Open the Project
Open your terminal (PowerShell, Command Prompt, or Git Bash) in this project folder:
```bash
cd Spam_Mail_Detection
```

### 2. Create and Activate a Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
  *(If you get an execution policy error, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

- **Mac/Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
Run the training script to process the dataset and generate the trained model artifacts:
```bash
python train.py
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🚀 Step-by-Step Guide: Pushing to GitHub

Follow these steps in your terminal to push this project to your GitHub account:

### Step 1: Create a New Repository on GitHub
1. Log in to [GitHub](https://github.com/).
2. Click the **`+`** icon in the top right and select **New repository**.
3. Name it (e.g. `spam-mail-detection`).
4. Set it to **Public**.
5. **Do NOT** check "Add a README file" or "Add .gitignore" (we already have them).
6. Click **Create repository**.

### Step 2: Initialize Git in your project folder
Run the following commands in your project terminal:
```bash
git init
```

### Step 3: Stage and Commit the Files
```bash
git add .
git commit -m "Initial commit: Spam mail detection using SVM and Streamlit"
```

### Step 4: Link to Your GitHub Repository and Push
*(Replace `<your-username>` with your actual GitHub username)*
```bash
git branch -M main
git remote add origin https://github.com/<your-username>/spam-mail-detection.git
git push -u origin main
```

---

## 🌐 Free Deployment (Streamlit Community Cloud)

Once pushed to GitHub, you can deploy your application live to the web for free:

1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **New app**.
3. Select your repository (`<your-username>/spam-mail-detection`), branch (`main`), and main file path (`app.py`).
4. Click **Deploy!**
5. Your web app will be live with a public URL in less than 2 minutes!

---

## 📊 Model & Methodology

- **Vectorization**: `TfidfVectorizer` (term frequency-inverse document frequency) with top 5,000 features, unigram + bigram support, and English stop words removed.
- **Classifier**: Support Vector Machine (`SVC`) with a linear kernel and calibrated probability estimates.
- **Evaluation Metrics**:
  - Accuracy
  - Precision & Recall
  - Confusion Matrix (monitoring False Positives)
