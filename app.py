
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION AND PATHS
# ============================================================

st.set_page_config(
    page_title="BankIQ | Customer Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Models"
DATA_PATHS = [BASE_DIR / "data" / "bank.csv", BASE_DIR / "bank.csv"]

MODEL_FILES = {
    "model": "bank_deposit_final_model.joblib",
    "scaler": "bank_deposit_scaler.joblib",
    "label_encoder": "bank_deposit_label_encoder.joblib",
    "preprocessing_info": "bank_deposit_preprocessing_info.joblib",
    "threshold": "bank_deposit_threshold.joblib",
    "metrics": "bank_deposit_metrics.joblib",
    "feature_importance": "bank_deposit_feature_importance.joblib",
}

# Optional evaluation artifacts. The app will use them only if present.
OPTIONAL_FILES = {
    "confusion_matrix": "bank_deposit_confusion_matrix.joblib",
    "error_analysis": "bank_deposit_error_analysis.joblib",
    "test_predictions": "bank_deposit_test_predictions.joblib",
}


# ============================================================
# 2. THEME
# ============================================================

st.markdown(
    """
    <style>
    .stApp { background: #0B0B0B; color: #D0D0D0; }
    .main .block-container { padding-top: 1.8rem; padding-bottom: 2.5rem; }
    [data-testid="stSidebar"] { background: #080808; border-right: 1px solid #292929; }
    [data-testid="stSidebar"] * { color: #C8C8C8 !important; }
    h1, h2, h3, h4 { color: #E0E0E0 !important; }
    p, label, li, span { color: #C0C0C0; }
    .main-title { font-size: 2.1rem; font-weight: 750; color: #EEEEEE; margin-bottom: .2rem; }
    .subtitle { color: #999999; font-size: 1rem; margin-bottom: 1.2rem; }
    [data-testid="stMetric"] {
        background: #171717; border: 1px solid #303030;
        padding: 14px; border-radius: 10px;
    }
    [data-testid="stMetricLabel"] { color: #A0A0A0 !important; }
    [data-testid="stMetricValue"] { color: #EEEEEE !important; }
    .stTextInput input, .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        background: #202020 !important; color: #E0E0E0 !important;
        border: 1px solid #414141 !important;
    }
    div.stButton > button, div.stFormSubmitButton > button {
        background: #333333; color: #FFFFFF !important;
        border: 1px solid #555555; border-radius: 8px;
        font-weight: 600; padding: .55rem 1rem;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background: #505050; border-color: #777777;
    }
    button[data-baseweb="tab"] { color: #AAAAAA !important; }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FFFFFF !important; border-bottom-color: #BBBBBB !important;
    }
    [data-testid="stExpander"], [data-testid="stForm"] {
        background: #141414; border: 1px solid #303030; border-radius: 10px;
    }
    [data-testid="stDataFrame"] { border: 1px solid #303030; border-radius: 8px; }
    hr { border-color: #303030 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_model_artifacts():
    """Load the model and preprocessing artifacts once per app session."""
    missing = [
        filename for filename in MODEL_FILES.values()
        if not (MODEL_DIR / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s) in the Models folder: "
            + ", ".join(missing)
        )

    loaded = {
        key: joblib.load(MODEL_DIR / filename)
        for key, filename in MODEL_FILES.items()
    }
    for key, filename in OPTIONAL_FILES.items():
        path = MODEL_DIR / filename
        loaded[key] = joblib.load(path) if path.is_file() else None

    loaded["threshold"] = float(loaded["threshold"])
    loaded["feature_columns"] = loaded["preprocessing_info"]["feature_columns"]
    loaded["cat_cols"] = loaded["preprocessing_info"]["cat_cols"]
    loaded["num_cols"] = loaded["preprocessing_info"]["num_cols"]
    return loaded


@st.cache_data
def load_dataset():
    """Load the original CSV if it exists."""
    for path in DATA_PATHS:
        if path.is_file():
            return pd.read_csv(path)
    return None


try:
    artifacts = load_model_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info(
        "Check that the folder is named **Models** and contains all seven "
        "saved `.joblib` files listed in the project README."
    )
    st.stop()
except Exception:
    st.error("The saved model files could not be loaded.")
    st.info(
        "Check that the model files were created with compatible library "
        "versions and that they are not corrupted."
    )
    st.stop()

model = artifacts["model"]
scaler = artifacts["scaler"]
label_encoder = artifacts["label_encoder"]
preprocessing_info = artifacts["preprocessing_info"]
threshold = artifacts["threshold"]
metrics = artifacts["metrics"]
feature_importance = artifacts["feature_importance"]
feature_columns = artifacts["feature_columns"]
cat_cols = artifacts["cat_cols"]
num_cols = artifacts["num_cols"]

dataset = load_dataset()


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def get_yes_probability(probabilities):
    """Find the probability column corresponding to the original 'yes' label."""
    classes = list(getattr(model, "classes_", []))
    encoded_yes = None

    try:
        yes_position = list(label_encoder.classes_).index("yes")
        encoded_yes = yes_position
    except (AttributeError, ValueError):
        pass

    if classes and encoded_yes in classes:
        return float(probabilities[classes.index(encoded_yes)])

    # Fallback for models trained directly on string labels.
    if classes and "yes" in classes:
        return float(probabilities[classes.index("yes")])

    # Last fallback matches the class order used by LabelEncoder.
    if encoded_yes is not None and encoded_yes < len(probabilities):
        return float(probabilities[encoded_yes])

    raise ValueError("Could not identify the probability for the 'yes' class.")


def make_prediction(input_data):
    """Apply the same encoding, column alignment and scaling used during training."""
    encoded = pd.get_dummies(
        input_data,
        columns=cat_cols,
        dtype=int,
        drop_first=True,
    )
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    missing_numeric = [col for col in num_cols if col not in encoded.columns]
    if missing_numeric:
        raise ValueError(
            "The following numeric model features are missing: "
            + ", ".join(missing_numeric)
        )

    encoded[num_cols] = scaler.transform(encoded[num_cols])


    if not hasattr(model, "predict_proba"):
        raise ValueError("The saved model does not support probability predictions.")

    probabilities = model.predict_proba(encoded)[0]
    yes_probability = get_yes_probability(probabilities)
    prediction = int(yes_probability >= threshold)
    return prediction, yes_probability


def feature_importance_frame(raw_importance, feature_names):
    """Convert common saved feature-importance formats to a tidy DataFrame."""
    if raw_importance is None:
        return pd.DataFrame(columns=["Feature", "Importance"])

    if isinstance(raw_importance, pd.DataFrame):
        frame = raw_importance.copy()
        if {"Feature", "Importance"}.issubset(frame.columns):
            return frame[["Feature", "Importance"]].copy()
        if frame.shape[1] >= 1:
            frame = frame.reset_index()
            frame.columns = ["Feature", "Importance"] + list(frame.columns[2:])
            return frame[["Feature", "Importance"]]

    if isinstance(raw_importance, pd.Series):
        frame = raw_importance.rename("Importance").rename_axis("Feature").reset_index()
        return frame[["Feature", "Importance"]]

    if isinstance(raw_importance, dict):
        # Supports either {feature: importance} or {'feature_importances': array}.
        for key in ("feature_importances", "importances", "importance"):
            if key in raw_importance:
                values = np.asarray(raw_importance[key]).ravel()
                names = raw_importance.get("feature_names", feature_names)
                if len(names) == len(values):
                    return pd.DataFrame({"Feature": names, "Importance": values})
        if raw_importance and all(np.isscalar(v) for v in raw_importance.values()):
            return pd.DataFrame(
                {"Feature": list(raw_importance.keys()),
                 "Importance": list(raw_importance.values())}
            )

    values = np.asarray(raw_importance).ravel()
    if len(values) == len(feature_names):
        return pd.DataFrame({"Feature": feature_names, "Importance": values})

    return pd.DataFrame(columns=["Feature", "Importance"])


def metric_value(metric_name, default=None):
    """Read a metric while tolerating common key naming variations."""
    if not isinstance(metrics, dict):
        return default
    aliases = {
        "accuracy": ["Accuracy", "accuracy"],
        "precision": ["Precision", "precision"],
        "recall": ["Recall", "recall"],
        "f1": ["F1 Score", "F1", "f1", "f1_score"],
        "roc_auc": ["ROC-AUC", "ROC_AUC", "roc_auc", "ROC AUC"],
    }
    for key in aliases.get(metric_name, [metric_name]):
        if key in metrics:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                return metrics[key]
    return default


def show_confusion_matrix():
    """Display a saved confusion matrix if the project has one."""
    cm = artifacts.get("confusion_matrix")
    if cm is None and isinstance(metrics, dict):
        cm = metrics.get("Confusion Matrix", metrics.get("confusion_matrix"))

    if cm is None:
        st.info(
            "A confusion matrix is not included in the saved evaluation artifacts. "
            "To display it here, save the held-out test confusion matrix as "
            "`Models/bank_deposit_confusion_matrix.joblib`."
        )
        return

    cm = np.asarray(cm)
    if cm.shape != (2, 2):
        st.warning("The saved confusion matrix is not a 2 × 2 classification matrix.")
        return

    fig, ax = plt.subplots(figsize=(5.5, 4))
    fig.patch.set_facecolor("#171717")
    ax.set_facecolor("#171717")
    image = ax.imshow(cm, cmap="Greys")
    ax.set_xticks([0, 1], labels=["Predicted No", "Predicted Yes"])
    ax.set_yticks([0, 1], labels=["Actual No", "Actual Yes"])
    ax.set_title("Confusion Matrix", color="#E0E0E0")
    ax.tick_params(colors="#D0D0D0")
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, str(value), ha="center", va="center",
                color="#111111" if value > cm.max() / 2 else "#FFFFFF",
                fontsize=13, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color("#555555")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ============================================================
# 5. SIDEBAR AND HEADER
# ============================================================

with st.sidebar:
    st.markdown("## 🏦 BankIQ")
    st.caption("CUSTOMER INTELLIGENCE PLATFORM")
    st.markdown("---")
    st.markdown("### Model Information")
    st.write("**Algorithm:** Soft Voting Classifier")
    st.write("**Base estimators:** 4")
    st.write(f"**Decision threshold:** {threshold:.3f}")
    st.markdown("---")
    st.caption("Bank Term Deposit Prediction")
    st.caption("Machine Learning Portfolio Project")

st.markdown(
    '<div class="main-title">Banking Customer Intelligence</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Estimate a customer’s likelihood of subscribing '
    'to a bank term deposit using an ensemble machine learning model.</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

tab_predict, tab_insights, tab_performance, tab_about = st.tabs(
    ["🔮 Customer Prediction", "📊 Data Insights", "📈 Model Performance", "ℹ️ About"]
)


# ============================================================
# 6. CUSTOMER PREDICTION
# ============================================================

with tab_predict:
    st.subheader("Customer Information")
    st.write(
        "Enter the customer's demographic, banking and campaign details. "
        "Fields are initialized with reasonable example values; change them "
        "to match the customer being assessed."
    )

    with st.form("customer_prediction_form"):
        personal_col, banking_col = st.columns(2)

        with personal_col:
            st.markdown("#### 👤 Personal Information")
            age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
            job = st.selectbox(
                "Occupation",
                ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                 "retired", "self-employed", "services", "student", "technician",
                 "unemployed", "unknown"],
                index=4,
            )
            marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
            education = st.selectbox(
                "Education", ["secondary", "tertiary", "primary", "unknown"], index=0
            )
            balance = st.number_input(
                "Account Balance", min_value=-100000, max_value=1000000,
                value=1000, step=100,
            )

        with banking_col:
            st.markdown("#### 🏦 Banking and Campaign Information")
            default = st.selectbox("Credit in Default", ["no", "yes"])
            housing = st.selectbox("Housing Loan", ["no", "yes"])
            loan = st.selectbox("Personal Loan", ["no", "yes"])
            contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
            day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15)
            month = st.selectbox(
                "Last Contact Month",
                ["jan", "feb", "mar", "apr", "may", "jun",
                 "jul", "aug", "sep", "oct", "nov", "dec"],
                index=4,
            )
            campaign = st.number_input(
                "Contacts in Current Campaign", min_value=1, max_value=100,
                value=2, step=1,
            )
            pdays = st.number_input(
                "Days Since Previous Contact (-1 = Never)",
                min_value=-1, max_value=999, value=-1, step=1,
            )
            previous = st.number_input(
                "Contacts Before This Campaign", min_value=0, max_value=100,
                value=0, step=1,
            )
            poutcome = st.selectbox(
                "Previous Campaign Outcome", ["unknown", "success", "failure", "other"]
            )

        submitted = st.form_submit_button(
            "🔮 Predict Subscription", use_container_width=True
        )

    if submitted:
        try:
            customer = pd.DataFrame([{
                "age": int(age),
                "job": job,
                "marital": marital,
                "education": education,
                "default": default,
                "balance": float(balance),
                "housing": housing,
                "loan": loan,
                "contact": contact,
                "day": int(day),
                "month": month,
                "campaign": int(campaign),
                "pdays": int(pdays),
                "previous": int(previous),
                "poutcome": poutcome,
            }])

            required_input_columns = set(feature_columns)
            raw_expected = set(cat_cols) | set(num_cols)
            missing_input_columns = raw_expected - set(customer.columns)
            if missing_input_columns:
                raise ValueError(
                    "The input form does not provide all model fields: "
                    + ", ".join(sorted(missing_input_columns))
                )

            prediction, yes_probability = make_prediction(customer)
            probability_percent = yes_probability * 100

            if probability_percent >= 70:
                lead_category = "High Potential"
                lead_action = "Prioritise this customer for a marketing campaign."
            elif probability_percent >= 40:
                lead_category = "Medium Potential"
                lead_action = "Consider follow-up communication."
            else:
                lead_category = "Low Potential"
                lead_action = "Low priority for the current campaign."

            st.markdown("---")
            st.subheader("Prediction Results")

            result_col1, result_col2 = st.columns(2)
            with result_col1:
                if prediction == 1:
                    st.success("YES — Likely to Subscribe")
                else:
                    st.warning("NO — Unlikely to Subscribe")
                st.metric("Probability of Subscription", f"{probability_percent:.2f}%")
                st.progress(float(np.clip(yes_probability, 0, 1)))
                st.caption(f"Model decision threshold: {threshold:.3f}")

            with result_col2:
                st.markdown("#### Probability Breakdown")
                st.metric("Predicted Class", "Yes" if prediction == 1 else "No")
                st.metric("Probability of No", f"{(1 - yes_probability) * 100:.2f}%")
                st.metric("Probability of Yes", f"{probability_percent:.2f}%")

            st.markdown("#### Customer Lead Scoring")
            score_col1, score_col2 = st.columns(2)
            score_col1.metric("Customer Lead Score", f"{probability_percent:.1f}/100")
            score_col2.metric("Lead Category", lead_category)
            st.write("**Recommended marketing action:**", lead_action)

            if lead_category == "High Potential":
                st.success("HIGH POTENTIAL CUSTOMER")
            elif lead_category == "Medium Potential":
                st.warning("MEDIUM POTENTIAL CUSTOMER")
            else:
                st.info("LOW POTENTIAL CUSTOMER")

            st.caption(
                "The lead categories (40% and 70%) are presentation bands. "
                "The actual Yes/No prediction is determined by the saved model threshold."
            )

            st.markdown("---")
            st.subheader("Why this prediction?")
            importance_df = feature_importance_frame(feature_importance, feature_columns)
            if not importance_df.empty:
                importance_df["Importance"] = pd.to_numeric(
                    importance_df["Importance"], errors="coerce"
                )
                importance_df = importance_df.dropna(subset=["Importance"])
                importance_df = importance_df.sort_values(
                    "Importance", ascending=False
                ).head(10).sort_values("Importance", ascending=True)

                if not importance_df.empty:
                    fig, ax = plt.subplots(figsize=(9, 5))
                    fig.patch.set_facecolor("#171717")
                    ax.set_facecolor("#171717")
                    ax.barh(
                        importance_df["Feature"].astype(str),
                        importance_df["Importance"],
                        color="#BDBDBD",
                    )
                    ax.set_xlabel("Feature Importance", color="#D0D0D0")
                    ax.set_title("Top 10 Global Feature Importance", color="#E0E0E0")
                    ax.tick_params(colors="#D0D0D0")
                    for spine in ax.spines.values():
                        spine.set_color("#444444")
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    st.caption(
                        "These are global feature-importance values: they describe "
                        "overall model behaviour, not the reason for this individual "
                        "customer's prediction. Feature importance does not show whether "
                        "a feature increases or decreases subscription probability."
                    )
                else:
                    st.info("The saved feature-importance file contains no usable numeric values.")
            else:
                st.info(
                    "No supported feature-importance format was found in the saved artifact. "
                    "The model prediction is available, but this explanation chart needs "
                    "a feature-importance artifact."
                )

        except (ValueError, KeyError, TypeError, IndexError) as exc:
            st.error("We couldn't generate a prediction from these inputs.")
            st.info(f"Please check the entered values and saved model configuration. Details: {exc}")
        except Exception:
            st.error("An unexpected issue prevented the prediction.")
            st.info(
                "Please check that the model artifacts match the preprocessing used "
                "during training, then try again. No raw error trace is shown in the app."
            )


# ============================================================
# 7. DATA INSIGHTS
# ============================================================

with tab_insights:
    st.subheader("Customer Data Insights")
    st.write(
        "Explore customer demographics, campaign behaviour and subscription "
        "patterns in the original bank marketing dataset."
    )

    if dataset is None:
        st.warning(
            "The dataset was not found. Place `bank.csv` inside the `data` folder "
            "or beside `app.py` to enable this section."
        )
    elif dataset.empty:
        st.warning("The dataset file is present but contains no rows.")
    elif "deposit" not in dataset.columns:
        st.error("The dataset does not contain the expected target column: `deposit`.")
    else:
        target = dataset["deposit"].astype(str).str.strip().str.lower()
        yes_count = int(target.eq("yes").sum())
        no_count = int(target.eq("no").sum())
        total = len(dataset)
        subscription_rate = (yes_count / total * 100) if total else 0

        st.markdown("#### Customer Overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Customers", f"{total:,}")
        m2.metric("Subscribed", f"{yes_count:,}")
        m3.metric("Not Subscribed", f"{no_count:,}")
        m4.metric("Subscription Rate", f"{subscription_rate:.2f}%")

        st.markdown("---")
        st.markdown("#### Target Class Distribution")
        left, right = st.columns(2)
        with left:
            counts = target.value_counts().rename_axis("Subscription").to_frame("Customers")
            st.bar_chart(counts)
        with right:
            if total:
                percentages = (target.value_counts(normalize=True) * 100)
                percentages = percentages.rename_axis("Subscription").to_frame("Percentage")
                st.bar_chart(percentages)
        if total:
            majority_share = max(yes_count, no_count) / total * 100
            st.caption(
                f"The larger class represents {majority_share:.1f}% of the data. "
                "This is why precision, recall, F1 and ROC-AUC should be considered "
                "alongside accuracy for classification evaluation."
            )

        st.markdown("---")
        st.markdown("#### Dataset Quality")
        q1, q2, q3 = st.columns(3)
        q1.metric("Duplicate Rows", f"{int(dataset.duplicated().sum()):,}")
        q2.metric("Columns", f"{dataset.shape[1]:,}")
        q3.metric("Missing Cells", f"{int(dataset.isna().sum().sum()):,}")

        missing = dataset.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values were detected by pandas.")
        else:
            missing_df = pd.DataFrame({
                "Feature": missing.index,
                "Missing Values": missing.values,
                "Missing Percentage": (missing.values / len(dataset) * 100),
            })
            st.dataframe(
                missing_df.style.format({"Missing Percentage": "{:.2f}%"}),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")
        categorical_candidates = [
            col for col in dataset.select_dtypes(include=["object", "category"]).columns
            if col != "deposit"
        ]
        numerical_candidates = list(
            dataset.select_dtypes(include=np.number).columns
        )

        st.markdown("#### Categorical Feature Analysis")
        if categorical_candidates:
            selected_cat = st.selectbox(
                "Choose a categorical feature", categorical_candidates,
                key="insights_categorical",
            )
            cat_data = dataset[[selected_cat]].copy()
            cat_data["Subscription"] = target
            cat_data["Subscribed"] = target.eq("yes").astype(int)
            summary = (
                cat_data.groupby(selected_cat, dropna=False, observed=False)
                .agg(
                    Customers=("Subscribed", "size"),
                    Subscribers=("Subscribed", "sum"),
                    Subscription_Rate=("Subscribed", "mean"),
                )
                .reset_index()
            )
            summary["Subscription_Rate"] *= 100
            summary = summary.sort_values("Subscription_Rate", ascending=False)
            c1, c2 = st.columns(2)
            with c1:
                st.write("Subscription rate by category (%)")
                st.bar_chart(summary.set_index(selected_cat)["Subscription_Rate"])
            with c2:
                st.write("Customer count by category")
                st.bar_chart(summary.set_index(selected_cat)["Customers"])
            st.dataframe(
                summary.rename(columns={
                    selected_cat: "Category",
                    "Customers": "Total Customers",
                    "Subscribers": "Subscribed Customers",
                    "Subscription_Rate": "Subscription Rate (%)",
                }).style.format({"Subscription Rate (%)": "{:.2f}%"}),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Compare both the subscription rate and customer count: a high rate "
                "in a very small group may represent fewer potential customers overall."
            )
        else:
            st.info("No categorical columns were found.")

        st.markdown("---")
        st.markdown("#### Numerical Feature Analysis")
        if numerical_candidates:
            selected_num = st.selectbox(
                "Choose a numerical feature", numerical_candidates,
                key="insights_numerical",
            )
            plot_data = dataset[[selected_num]].copy()
            plot_data["Subscription"] = target
            fig, ax = plt.subplots(figsize=(9, 4.5))
            fig.patch.set_facecolor("#171717")
            ax.set_facecolor("#171717")
            for label, group in plot_data.groupby("Subscription"):
                values = pd.to_numeric(group[selected_num], errors="coerce").dropna()
                ax.hist(values, bins=30, alpha=0.55, label=str(label))
            ax.set_title(f"{selected_num} distribution by subscription", color="#E0E0E0")
            ax.set_xlabel(selected_num, color="#C0C0C0")
            ax.set_ylabel("Number of customers", color="#C0C0C0")
            ax.tick_params(colors="#C0C0C0")
            ax.legend(facecolor="#222222", labelcolor="#E0E0E0")
            for spine in ax.spines.values():
                spine.set_color("#444444")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.dataframe(
                dataset[numerical_candidates].describe().T,
                use_container_width=True,
            )
            if selected_num == "duration":
                st.info(
                    "Duration is shown for exploratory analysis. If it was excluded "
                    "from model inputs to avoid information unavailable before a call, "
                    "keep it excluded from the deployed prediction form."
                )
        else:
            st.info("No numerical columns were found.")

        st.markdown("---")
        st.markdown("#### Dataset Preview")
        with st.expander("View first 20 rows"):
            st.dataframe(dataset.head(20), use_container_width=True)


# ============================================================
# 8. MODEL PERFORMANCE
# ============================================================

with tab_performance:
    st.subheader("Model Performance Dashboard")
    st.write(
        "The following values are loaded from the saved evaluation metrics "
        "artifact. They should represent the held-out test set, not training data."
    )

    accuracy = metric_value("accuracy")
    precision = metric_value("precision")
    recall = metric_value("recall")
    f1 = metric_value("f1")
    roc_auc = metric_value("roc_auc")

    metric_items = [
        ("Accuracy", accuracy, True),
        ("Precision", precision, True),
        ("Recall", recall, True),
        ("F1 Score", f1, True),
        ("ROC-AUC", roc_auc, False),
    ]
    metric_cols = st.columns(5)
    for col, (label, value, percent) in zip(metric_cols, metric_items):
        if value is None:
            col.metric(label, "Not saved")
        elif percent:
            col.metric(label, f"{value * 100:.2f}%")
        else:
            col.metric(label, f"{value:.4f}")

    st.markdown("---")
    st.markdown("#### Classification Threshold")
    st.metric("Optimised Decision Threshold", f"{threshold:.3f}")
    st.write(
        "The threshold converts the model's predicted subscription probability "
        "into a Yes/No class. A lower threshold generally identifies more possible "
        "subscribers but can also increase false positives. The chosen threshold "
        "should be justified using validation data and the project's business goal."
    )

    st.markdown("---")
    st.markdown("#### Confusion Matrix")
    show_confusion_matrix()

    st.markdown("---")
    st.markdown("#### Ensemble Architecture")
    st.write(
        "The deployed model is a Soft Voting Classifier. According to the current "
        "project configuration, it combines four estimators:"
    )
    st.markdown(
        """
        1. Logistic Regression
        2. Decision Tree
        3. Random Forest
        4. XGBoost

        Soft voting combines the class probabilities produced by the estimators
        before applying the final decision threshold.
        """
    )

    st.markdown("---")
    st.markdown("#### Error Analysis")
    error_analysis = artifacts.get("error_analysis")
    test_predictions = artifacts.get("test_predictions")

    if error_analysis is not None:
        if isinstance(error_analysis, pd.DataFrame):
            st.dataframe(error_analysis, use_container_width=True)
        elif isinstance(error_analysis, dict):
            for name, value in error_analysis.items():
                st.markdown(f"**{name.replace('_', ' ').title()}**")
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    st.dataframe(value, use_container_width=True)
                else:
                    st.write(value)
        else:
            st.write(error_analysis)
    elif test_predictions is not None:
        st.write("A test-prediction artifact is available:")
        if isinstance(test_predictions, pd.DataFrame):
            st.dataframe(test_predictions.head(100), use_container_width=True)
        else:
            st.write(test_predictions)
        st.caption(
            "For group-level error analysis, save the true labels, predicted labels "
            "and relevant customer features from the held-out test set."
        )
    else:
        st.info(
            "No held-out test error-analysis artifact was found. This app does not "
            "calculate errors on the training dataset and present them as test results. "
            "To complete this analysis, export test-set actual labels, predicted labels "
            "and customer features to `Models/bank_deposit_error_analysis.joblib`."
        )

    st.caption(
        "Accuracy alone can be misleading when the target classes are imbalanced. "
        "Precision, recall, F1 and ROC-AUC provide additional information about "
        "false positives, missed subscribers and ranking quality."
    )


# ============================================================
# 9. ABOUT PROJECT
# ============================================================

with tab_about:
    st.subheader("About BankIQ")
    st.write(
        """
        **Bank Term Deposit Subscription Prediction** is a supervised machine
        learning classification project. It estimates whether a bank customer
        is likely to subscribe to a term deposit based on demographic, banking
        and marketing campaign information.
        """
    )

    st.markdown("---")
    st.markdown("#### Problem and Intended Use")
    st.write(
        "The application is designed to help a bank marketing team prioritize "
        "customers for follow-up. Its output is a model estimate, not a guarantee "
        "of customer behaviour or a substitute for human review."
    )

    st.markdown("---")
    st.markdown("#### Project Workflow")
    st.markdown(
        """
        1. Problem definition
        2. Data collection and exploratory data analysis
        3. Data cleaning and preprocessing
        4. Categorical encoding and numerical scaling
        5. Baseline model
        6. Model comparison
        7. Cross-validation and hyperparameter tuning
        8. Model evaluation using classification metrics
        9. Feature interpretation and error analysis
        10. Saving the trained model and preprocessing artifacts
        11. Streamlit deployment
        """
    )

    st.markdown("---")
    st.markdown("#### How to Use the Application")
    st.markdown(
        """
        1. Open **Customer Prediction**.
        2. Enter the customer's information in the form.
        3. Select **Predict Subscription**.
        4. Review the predicted class, probability and lead category.
        5. Review the global feature-importance chart.
        6. Use **Data Insights** to explore the source dataset.
        7. Use **Model Performance** to review saved evaluation results.
        """
    )

    st.markdown("---")
    st.markdown("#### Technologies Used")
    st.markdown(
        """
        - Python
        - Pandas and NumPy
        - Scikit-learn and XGBoost
        - Matplotlib
        - Streamlit
        - Joblib
        """
    )

    st.markdown("---")
    st.markdown("#### Important Limitations")
    st.write(
        "Predictions depend on the data and model used during training. "
        "Feature importance is a global explanation and does not establish "
        "causation or explain an individual prediction. Model performance can "
        "change when customer behaviour or campaign practices change."
    )

    st.caption("BankIQ | Machine Learning Portfolio Project")


# ============================================================
# 10. FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "BankIQ | Bank Term Deposit Prediction | "
    "Predictions are estimates and should be reviewed by a human decision-maker."
)
