import streamlit as st
import joblib

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Smart SMS Spam Detector",
    page_icon="📧",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("svm_spam_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ---------------- CSS ---------------- #

st.markdown("""
<style>

.main{
    background-color:#0B1020;
}

.stButton > button{
    width:100%;
    height:60px;
    border-radius:12px;
    border:none;
    background:linear-gradient(90deg,#7C3AED,#A855F7);
    color:white;
    font-size:20px;
    font-weight:bold;
}

.stButton > button:hover{
    transform:scale(1.02);
}

.result-card{
    padding:25px;
    border-radius:15px;
    color:white;
    text-align:center;
}

.safe{
    background:linear-gradient(90deg,#059669,#10B981);
}

.spam{
    background:linear-gradient(90deg,#DC2626,#EF4444);
}

.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("🛡️ Message Security Center")

    st.info("""
### Smart SMS Spam Detector

Analyze text messages and identify whether they are legitimate messages or spam using machine learning and NLP.
""")

    st.markdown("""
### 🔍 Capabilities

- Spam Detection
- Text Classification
- NLP Processing
- Real-Time Analysis
- Message Security Assessment

### 🧠 Model

- TF-IDF Vectorization
- Linear Support Vector Machine
- Hyperparameter Tuned
""")

    st.warning("""
⚠️ Educational Project

This application is intended for learning, research and portfolio purposes.
""")

    st.markdown("---")

    st.write("👩‍💻 Developed by")
    st.write("**Kalyani M G**")

# ---------------- HEADER ---------------- #

st.markdown("""
<h1 style='text-align:center;color:#A855F7;'>
📧 Smart SMS Spam Detector
</h1>

<p style='text-align:center;font-size:20px;color:#BDBDBD;'>
AI-Powered Message Security System
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- DASHBOARD CARDS ---------------- #

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Model", "Linear SVM")

with c2:
    st.metric("Technique", "NLP")

with c3:
    st.metric("Status", "Active")

st.markdown("---")

# ---------------- INPUT ---------------- #

st.subheader("✉️ Enter SMS Message")

message = st.text_area(
    "",
    height=220,
    placeholder="Type or paste a message here..."
)

# ---------------- PREDICTION ---------------- #

if st.button("🔍 Analyze Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        transformed = vectorizer.transform([message])

        prediction = model.predict(transformed)[0]
        st.write("Prediction Value:", prediction)

        st.markdown("---")

        st.subheader("📊 Analysis Result")

        if prediction == 1:

            st.markdown("""
            <div class="result-card spam">
                <h2>🚨 SPAM DETECTED</h2>
                <p>
                This message contains patterns commonly associated
                with spam, promotions or fraudulent communication.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.error("""
Potential Indicators:

• Suspicious offers
• Urgent action requests
• Promotional content
• Unknown sender behaviour
""")

        else:

            st.markdown("""
            <div class="result-card safe">
                <h2>✅ SAFE MESSAGE</h2>
                <p>
                This message appears to be a legitimate personal
                or business communication.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.success("""
Message Assessment:

• No major spam indicators detected
• Appears legitimate
• Safe communication pattern
""")

# ---------------- PROJECT INFO ---------------- #

st.markdown("---")

with st.expander("📖 View Project Information"):

    st.write("""
### Smart SMS Spam Detector

This application uses Natural Language Processing (NLP) and Machine Learning techniques to classify SMS messages as Spam or Ham.

The system converts text messages into numerical features using TF-IDF Vectorization and applies a tuned Support Vector Machine (SVM) classifier for prediction.

### Technologies Used

- Python
- Streamlit
- Scikit-Learn
- TF-IDF Vectorization
- Linear SVM
- NLP

### Features

- Real-Time SMS Classification
- Spam Detection
- Text Analysis
- Interactive Security Dashboard
- Fast Prediction

### Dataset

SMS Spam Collection Dataset

### Developer

Kalyani M G
""")

st.markdown("---")

st.caption(
    "© 2026 Smart SMS Spam Detector | Developed by Kalyani M G"
)
