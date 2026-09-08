import os
import joblib
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Spam Mail Detector | AI & ML",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for a modern portfolio look
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.55rem 1.5rem;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        color: white;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    .result-card-spam {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border-left: 6px solid #DC2626;
        padding: 20px;
        border-radius: 10px;
        color: #991B1B;
        margin-top: 15px;
    }
    .result-card-ham {
        background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
        border-left: 6px solid #16A34A;
        padding: 20px;
        border-radius: 10px;
        color: #166534;
        margin-top: 15px;
    }
    .badge-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model_path = os.path.join("model", "spam_svm_model.pkl")
    vectorizer_path = os.path.join("model", "tfidf_vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_artifacts()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=70)
    st.markdown("### **Spam Mail Classifier**")
    st.markdown("**Algorithm**: Support Vector Machine (Linear SVM)")
    st.markdown("**Features**: TF-IDF N-grams (1, 2)")
    st.markdown("---")

    st.markdown("### 🧪 Quick Test Samples")
    st.caption("Click any sample below to load into the text box:")

    sample_project = (
        "Hi Lonnie,\n\n"
        "Just wanted to touch base regarding our project’s next steps. "
        "Please find the updated project plan attached. Let me know if you have any questions.\n\n"
        "Kind regards,\nTerry Griffin"
    )

    sample_spam_prize = (
        "CONGRATULATIONS! You have been selected as the official winner of our $5,000 Cash Prize! "
        "Call 1-800-555-CLAIM now or click the link to claim your reward immediately. "
        "Offer expires in 24 hours. T&Cs apply."
    )

    sample_spam_phish = (
        "URGENT: Your bank account access has been suspended due to suspicious activity. "
        "Please verify your credentials immediately by clicking here: http://secure-banking-verify.com/login "
        "Failure to verify within 12 hours will result in permanent account deactivation."
    )

    sample_casual = (
        "Hey, are we still meeting for lunch today at 1 PM? "
        "Let me know if you want me to pick up anything on the way!"
    )

    if st.button("📋 Sample 1: Project Meeting (Ham)"):
        st.session_state["input_text"] = sample_project
    if st.button("🎁 Sample 2: Prize / Lottery (Spam)"):
        st.session_state["input_text"] = sample_spam_prize
    if st.button("🚨 Sample 3: Urgent Phishing (Spam)"):
        st.session_state["input_text"] = sample_spam_phish
    if st.button("💬 Sample 4: Casual Note (Ham)"):
        st.session_state["input_text"] = sample_casual

    st.markdown("---")
    st.markdown("### ⚙️ Model Status")
    if model is not None and vectorizer is not None:
        st.success("✅ Model & Vectorizer Loaded")
    else:
        st.warning("⚠️ Model not found. Please run `python train.py` first.")

# Main App Header
st.markdown('<div class="main-header">🛡️ Email Spam Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Classify emails as Legitimate (Ham) or Spam in real time using Machine Learning.</div>', unsafe_allow_html=True)

# Main Text Input Area
default_input = st.session_state.get("input_text", "")
user_email = st.text_area(
    "Paste or type the email message content below:",
    value=default_input,
    height=200,
    placeholder="Enter email body text here..."
)

col1, col2, col3 = st.columns([1.2, 1, 3])

with col1:
    classify_btn = st.button("🔍 Analyze Email", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear Input", use_container_width=True):
        st.session_state["input_text"] = ""
        st.rerun()

# Prediction Logic
if classify_btn:
    if not user_email.strip():
        st.warning("⚠️ Please enter or paste some email text before analyzing.")
    elif model is None or vectorizer is None:
        st.error("❌ Model artifacts are not loaded. Run `python train.py` first to train and generate the model.")
    else:
        # Preprocess & Vectorize
        text_vectorized = vectorizer.transform([user_email])

        # Predict
        prediction = model.predict(text_vectorized)[0]

        # Calculate probabilities
        try:
            probabilities = model.predict_proba(text_vectorized)[0]
            ham_prob = probabilities[0] * 100
            spam_prob = probabilities[1] * 100
        except Exception:
            ham_prob = 0.0
            spam_prob = 100.0 if prediction == 1 else 0.0

        st.markdown("---")
        st.subheader("📊 Analysis Results")

        res_col1, res_col2 = st.columns([1.2, 1])

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card-spam">
                    <h3 style="margin-top:0; color:#DC2626;">🚨 SPAM DETECTED</h3>
                    <p style="font-size:1.05rem; margin-bottom:5px;">This email exhibits common spam or phishing characteristics.</p>
                    <b>Spam Confidence: {spam_prob:.2f}%</b>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-ham">
                    <h3 style="margin-top:0; color:#16A34A;">✅ LEGITIMATE (HAM)</h3>
                    <p style="font-size:1.05rem; margin-bottom:5px;">This email appears safe and normal with no strong spam triggers.</p>
                    <b>Legitimate Confidence: {ham_prob:.2f}%</b>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("#### Confidence Breakdown")
            st.write(f"**Legitimate (Ham):** {ham_prob:.1f}%")
            st.progress(int(ham_prob))
            st.write(f"**Spam:** {spam_prob:.1f}%")
            st.progress(int(spam_prob))

        # Keyword trigger detection
        common_triggers = [
            "free", "winner", "won", "prize", "cash", "claim", "urgent", "credit card",
            "bank", "account", "suspended", "password", "verify", "call now", "click here",
            "limited time", "offer", "reward", "lottery", "congratulations"
        ]
        found_triggers = [word for word in common_triggers if word in user_email.lower()]

        if found_triggers:
            st.markdown("#### 🚩 Highlighted Suspicious Keywords Found")
            chips = " ".join([f"<span class='badge-chip' style='background-color:#FEE2E2; color:#B91C1C;'>⚠️ {w}</span>" for w in found_triggers])
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("No standard high-risk trigger keywords detected in the text.")

# Footer
st.markdown("---")
st.caption("Built with Python, Scikit-Learn (SVM), and Streamlit.")
