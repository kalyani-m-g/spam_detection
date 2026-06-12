import streamlit as st
import joblib

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("svm_spam_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ---------------- CSS ---------------- #

st.markdown("""
<style>

.main {
    background-color: #F8FAFC;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    margin-top: 15px;
    color: white;
}

.spam {
    background: linear-gradient(90deg,#DC2626,#EF4444);
}

.safe {
    background: linear-gradient(90deg,#059669,#10B981);
}

.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 12px;
    background: linear-gradient(90deg,#2563EB,#7C3AED);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    transform: scale(1.02);
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("📩 Spam Detector")

    st.info("""
Detect whether an SMS message is Spam or Ham (Safe)
using Machine Learning and NLP.
""")

    st.markdown("""
### Features

✅ TF-IDF Vectorization

✅ Support Vector Machine (SVM)

✅ Real-time Message Analysis

✅ Spam Detection Dashboard
""")

    st.markdown("---")

    st.write("👩‍💻 Developed by")
    st.write("**Kalyani M G**")

# ---------------- HEADER ---------------- #

st.markdown("""
<h1 style='text-align:center;color:#7C3AED;'>
📩 Smart SMS Spam Detector
</h1>

<p style='text-align:center;font-size:18px;color:gray;'>
Machine Learning Powered Message Classification
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- EXAMPLES ---------------- #

col1, col2 = st.columns(2)

with col1:
    st.info("""
Example Spam:

Congratulations!

You have won ₹50,000.

Click here now to claim your reward.
""")

with col2:
    st.info("""
Example Safe Message:

Hi Kalyani,

Are we meeting tomorrow at 10 AM for the project discussion?
""")

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

        st.markdown("---")

        st.subheader("📊 Analysis Result")

        # Handles both numeric and text labels
        if str(prediction).lower() in ["spam", "1"]:

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

This application uses Natural Language Processing (NLP)
and Machine Learning to classify SMS messages as Spam or Ham.

### Technologies Used

- Python
- Streamlit
- Scikit-Learn
- TF-IDF Vectorization
- Support Vector Machine (SVM)

### Features

- Real-Time Spam Detection
- Interactive Dashboard
- NLP-Based Text Analysis
- User-Friendly Interface

### Developer

Kalyani M G
""")

st.markdown("---")

st.caption(
    "© 2026 Smart SMS Spam Detector | Developed by Kalyani M G"
)
