
# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ==========================================
# 2. LOAD SAVED MODELS AND PREPROCESSING FILES
# ==========================================

# Get project directory
BASE_DIR = Path(__file__).parent

# Define Models folder
MODEL_DIR = BASE_DIR / "Models"

# Load trained model
model = joblib.load(
    MODEL_DIR / "bank_deposit_final_model.joblib"
)

# Load scaler
scaler = joblib.load(
    MODEL_DIR / "bank_deposit_scaler.joblib"
)

# Load label encoder
label_encoder = joblib.load(
    MODEL_DIR / "bank_deposit_label_encoder.joblib"
)

# Load preprocessing information
preprocessing_info = joblib.load(
    MODEL_DIR / "bank_deposit_preprocessing_info.joblib"
)


# ==========================================
# 3. STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Bank Term Deposit Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)




st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    /* All headings in grey */
    h1, h2, h3 {
        color: #808080 !important;
    }

    /* Black metric boxes */
    div[data-testid="stMetric"] {
        background-color: #000000;
        border: 1px solid #333333;
        padding: 15px;
        border-radius: 10px;
    }

    /* Grey metric labels */
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label p {
        color: #b0b0b0 !important;
    }

    /* White metric numbers */
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    /* Grey metric delta text */
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #b0b0b0 !important;
    }

    /* Dark blue buttons */
    div.stButton > button {
        background-color: #17365D;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)



# Application title
st.title("🏦 Bank Term Deposit Prediction")

st.write(
    "Predict whether a customer is likely to subscribe "
    "to a bank term deposit."
)

st.divider()



# ==========================================
# 4. APPLICATION NAVIGATION
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict",
    "📊 Data Insights",
    "📈 Model Performance",
    "ℹ️ About"
])


# ==========================================
# 5. PREDICT SECTION
# ==========================================


# ==========================================
# 5. PREDICT SECTION
# ==========================================

with tab1:

    st.header("Customer Deposit Prediction")

    st.write(
        "Enter customer details to predict whether "
        "the customer is likely to subscribe to a term deposit."
    )

    st.divider()

    with st.form("prediction_form"):

        st.subheader("Customer Information")

        col1, col2, col3 = st.columns(3)

        # Numerical features
        with col1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=35
            )

            balance = st.number_input(
                "Account Balance",
                value=1000,
                step=100
            )

            day = st.number_input(
                "Last Contact Day",
                min_value=1,
                max_value=31,
                value=15
            )

            campaign = st.number_input(
                "Campaign Contacts",
                min_value=1,
                value=1
            )

        with col2:
            pdays = st.number_input(
                "Days Since Previous Contact",
                min_value=-1,
                value=-1
            )

            previous = st.number_input(
                "Previous Contacts",
                min_value=0,
                value=0
            )

            job = st.selectbox(
                "Job",
                [
                    "admin.", "technician", "services",
                    "management", "retired", "blue-collar",
                    "unemployed", "entrepreneur", "housemaid",
                    "unknown", "self-employed", "student"
                ]
            )

            marital = st.selectbox(
                "Marital Status",
                ["married", "single", "divorced"]
            )

            education = st.selectbox(
                "Education",
                ["secondary", "tertiary", "primary", "unknown"]
            )

        with col3:

            default = st.selectbox(
                "Credit in Default",
                ["no", "yes"]
            )

            housing = st.selectbox(
                "Housing Loan",
                ["yes", "no"]
            )

            loan = st.selectbox(
                "Personal Loan",
                ["no", "yes"]
            )

            contact = st.selectbox(
                "Contact Type",
                ["cellular", "telephone", "unknown"]
            )

            month = st.selectbox(
                "Last Contact Month",
                [
                    "may", "jun", "jul", "aug",
                    "oct", "nov", "dec", "jan",
                    "feb", "mar", "apr", "sep"
                ]
            )

            poutcome = st.selectbox(
                "Previous Campaign Outcome",
                ["unknown", "other", "failure", "success"]
            )

        st.divider()

        submit = st.form_submit_button(
            "Predict Deposit Subscription",
            use_container_width=True
        )

    # Prediction logic
    if submit:

        try:

            # Create input dataframe
            input_data = pd.DataFrame([{
                "age": age,
                "job": job,
                "marital": marital,
                "education": education,
                "default": default,
                "balance": balance,
                "housing": housing,
                "loan": loan,
                "contact": contact,
                "day": day,
                "month": month,
                "campaign": campaign,
                "pdays": pdays,
                "previous": previous,
                "poutcome": poutcome
            }])

            # Apply One-Hot Encoding
            input_encoded = pd.get_dummies(
                input_data,
                columns=preprocessing_info["cat_cols"],
                dtype=int,
                drop_first=True
            )

            # Match training feature columns
            input_encoded = input_encoded.reindex(
                columns=preprocessing_info["feature_columns"],
                fill_value=0
            )

            # Apply saved StandardScaler
            input_encoded[preprocessing_info["num_cols"]] = (
                scaler.transform(
                    input_encoded[preprocessing_info["num_cols"]]
                )
            )

            # Generate prediction
            prediction = model.predict(input_encoded)[0]

            # Decode prediction
            result = label_encoder.inverse_transform(
                [int(prediction)]
            )[0]

            # Get subscription probability
            classes = list(model.classes_)
            yes_encoded = int(label_encoder.transform(["yes"])[0])

            yes_index = classes.index(yes_encoded)

            probability = model.predict_proba(
                input_encoded
            )[0][yes_index]

            st.divider()

            st.subheader("Prediction Result")

            if result == "yes":

                st.success(
                    "The customer is likely to subscribe "
                    "to a term deposit!"
                )

            else:

                st.warning(
                    "The customer is unlikely to subscribe "
                    "to a term deposit."
                )

            st.metric(
                "Probability of Subscription",
                f"{probability * 100:.2f}%"
            )

        except Exception as e:

            st.error(
                "Unable to generate prediction. "
                "Please check the input values."
            )

            st.caption(f"Technical details: {e}")



# ==========================================
# 6. DATA INSIGHTS SECTION
# ==========================================


# ==========================================
# 6. DATA INSIGHTS SECTION
# ==========================================

with tab2:

    st.header("Bank Marketing Data Insights")

    st.write(
        "Explore customer information and understand "
        "the factors associated with term deposit subscriptions."
    )

    st.divider()

    # Load dataset
    @st.cache_data
    def load_data():

        return pd.read_csv(
            BASE_DIR / "data" / "bank.csv"
        )

    try:

        df = load_data()

        # Dataset Overview
        st.subheader("1. Dataset Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Customers", f"{len(df):,}")

        with col2:
            st.metric("Total Features", len(df.columns) - 1)

        with col3:
            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

        st.write("Dataset Preview")

        st.dataframe(df.head(10), use_container_width=True)

        st.divider()

        # Target Variable Distribution
        st.subheader("2. Term Deposit Subscription Distribution")

        target_counts = df["deposit"].value_counts()

        st.bar_chart(target_counts)

        st.write(
            "This chart shows the number of customers "
            "who subscribed and did not subscribe "
            "to a term deposit."
        )

        st.divider()

        # Age Distribution
        st.subheader("3. Customer Age Distribution")

        age_data = df["age"].value_counts(
            bins=10,
            sort=False
        )

        age_data.index = age_data.index.astype(str)

        st.bar_chart(age_data)

        st.write(
            "This chart displays the distribution "
            "of customers across different age groups."
        )

        st.divider()

        # Job-wise Subscription
        st.subheader("4. Job-wise Term Deposit Subscription")

        job_subscription = pd.crosstab(
            df["job"],
            df["deposit"]
        )

        st.bar_chart(job_subscription)

        st.write(
            "Compare subscription and non-subscription "
            "counts across different job categories."
        )

        st.divider()

        # Marital Status Analysis
        st.subheader("5. Subscription by Marital Status")

        marital_subscription = pd.crosstab(
            df["marital"],
            df["deposit"]
        )

        st.bar_chart(marital_subscription)

        st.divider()

        # Education Analysis
        st.subheader("6. Subscription by Education")

        education_subscription = pd.crosstab(
            df["education"],
            df["deposit"]
        )

        st.bar_chart(education_subscription)

    except FileNotFoundError:

        st.error(
            "Dataset not found. Please check that "
            "bank.csv is inside the data folder."
        )

    except Exception as e:

        st.error(f"Unable to load dataset: {e}")



# ==========================================
# 7. MODEL PERFORMANCE SECTION
# ==========================================

with tab3:
    st.header("📈 Model Performance")

    st.subheader("Final Model: Soft Voting Classifier")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", "73.35%")
    col2.metric("Precision", "78.06%")
    col3.metric("Recall", "60.87%")
    col4.metric("F1 Score", "68.40%")
    col5.metric("ROC-AUC", "78.49%")

    st.divider()

    st.subheader("Understanding the Metrics")

    st.write("""
    - **Accuracy:** Overall percentage of correct predictions.
    - **Precision:** Of customers predicted to subscribe, how many actually subscribed?
    - **Recall:** Of customers who actually subscribed, how many did the model identify?
    - **F1 Score:** Balance between precision and recall.
    - **ROC-AUC:** Measures how well the model distinguishes between the two classes.
    """)

    st.info(
        "The final model is a Soft Voting Classifier, "
        "which combines predictions from multiple models."
    )

# ==========================================
# 8. ABOUT SECTION
# ==========================================


with tab4:
    st.header("ℹ️ About This Project")

    st.subheader("Bank Term Deposit Prediction")

    st.write("""
    This machine learning project predicts whether a customer
    is likely to subscribe to a bank term deposit based on
    customer information and previous marketing campaign details.
    """)

    st.divider()

    st.subheader("🎯 Project Objective")

    st.write("""
    The main objective is to help banks identify potential
    customers who are more likely to subscribe to term deposits.

    This can help banks improve their marketing strategies
    and target customers more effectively.
    """)

    st.divider()

    st.subheader("⚙️ Machine Learning Workflow")

    steps = [
        "Exploratory Data Analysis (EDA)",
        "Data Preprocessing",
        "Model Training",
        "Cross-Validation",
        "Hyperparameter Tuning using GridSearchCV",
        "Model Evaluation",
        "Soft Voting Classifier",
        "Model Deployment using Streamlit"
    ]

    for i, step in enumerate(steps, start=1):
        st.write(f"{i}. {step}")

    st.divider()

    st.subheader("🛠️ Technologies Used")

    st.write("""
    - Python
    - Pandas and NumPy
    - Matplotlib and Seaborn
    - Scikit-learn
    - XGBoost
    - Joblib
    - Streamlit
    """)

    st.divider()

    st.subheader("📌 Important Note")

    st.info("""
    The customer call duration feature was excluded from
    model training because it is not known before a marketing
    call takes place. This helps avoid using information that
    would not be available when making a prediction.
    """)

    st.success("Thank you for exploring this project!")

