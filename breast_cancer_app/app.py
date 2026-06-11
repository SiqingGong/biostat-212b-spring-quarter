import streamlit as st
import pandas as pd
import joblib


# Load model and threshold
model = joblib.load("breast_cancer_survival_model.pkl")
threshold = joblib.load("threshold.pkl")

st.set_page_config(
    page_title="Breast Cancer Survival Risk Prediction",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            --rose: #c94f7c;
            --berry: #7f2d59;
            --ink: #263238;
            --muted: #667085;
            --soft: #f8edf3;
            --line: #ead5df;
            --good: #117a65;
            --bad: #b42318;
        }

        .stApp {
            background: linear-gradient(180deg, #fff8fb 0%, #ffffff 42%, #f7fbff 100%);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        .hero {
            display: grid;
            grid-template-columns: 1.25fr 0.75fr;
            gap: 1.5rem;
            align-items: stretch;
            padding: 1.6rem;
            border: 1px solid var(--line);
            border-radius: 18px;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.96), rgba(255,246,250,0.88)),
                url("https://images.unsplash.com/photo-1579154204601-01588f351e67?auto=format&fit=crop&w=1400&q=80");
            background-size: cover;
            background-position: center;
            box-shadow: 0 18px 48px rgba(127, 45, 89, 0.12);
            margin-bottom: 1.5rem;
            overflow: hidden;
        }

        .hero-copy {
            padding: 0.4rem 0;
        }

        .eyebrow {
            color: var(--berry);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .hero h1 {
            color: var(--ink);
            font-size: 2.8rem;
            line-height: 1.05;
            margin: 0 0 0.8rem 0;
        }

        .hero p {
            color: #475467;
            max-width: 680px;
            font-size: 1.03rem;
            line-height: 1.65;
            margin: 0;
        }

        .hero-panel {
            align-self: center;
            padding: 1.1rem;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(234, 213, 223, 0.95);
            backdrop-filter: blur(6px);
        }

        .hero-panel strong {
            display: block;
            color: var(--berry);
            font-size: 1.8rem;
            margin-bottom: 0.2rem;
        }

        .hero-panel span {
            color: var(--muted);
            font-size: 0.92rem;
        }

        .section-title {
            color: var(--berry);
            font-size: 1.35rem;
            font-weight: 800;
            margin: 1.35rem 0 0.7rem 0;
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }

        .section-title::before {
            content: "";
            width: 0.42rem;
            height: 1.45rem;
            border-radius: 999px;
            background: linear-gradient(180deg, #d65f8d, #6aa6a8);
        }

        .soft-note {
            padding: 0.85rem 1rem;
            border-left: 4px solid var(--rose);
            background: #fff4f8;
            color: #5b3445;
            border-radius: 10px;
            margin-bottom: 1.2rem;
        }

        .metric-card {
            min-height: 108px;
            padding: 1rem 1rem 0.95rem 1rem;
            border: 1px solid #efdce5;
            border-radius: 14px;
            background:
                linear-gradient(180deg, rgba(255, 246, 250, 0.96), rgba(255, 255, 255, 0.98)),
                #ffffff;
            box-shadow: 0 12px 26px rgba(127, 45, 89, 0.08);
            position: relative;
            overflow: hidden;
        }

        .metric-card::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 4px;
            background: linear-gradient(90deg, #c94f7c, #6aa6a8);
        }

        .metric-card .label {
            color: #8a5570;
            font-size: 0.8rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .metric-card .value {
            color: var(--berry);
            font-size: 1.7rem;
            font-weight: 850;
            margin-top: 0.2rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #efdce5;
            border-radius: 16px;
            background:
                linear-gradient(180deg, rgba(255, 248, 251, 0.88), rgba(255, 255, 255, 0.98));
            box-shadow: 0 12px 30px rgba(127, 45, 89, 0.07);
        }

        div[data-testid="stNumberInput"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSegmentedControl"] label {
            color: #7f2d59;
            font-weight: 700;
        }

        .field-label {
            color: #7f2d59;
            font-size: 0.92rem;
            font-weight: 750;
            margin: 0.55rem 0 0.28rem 0;
        }

        div[data-testid="stNumberInput"] input {
            color: #3f2b35;
            background-color: #fffafd;
            border-color: #ead5df;
        }

        div[data-baseweb="select"] > div {
            color: #3f2b35;
            background-color: #fffafd;
            border-color: #ead5df;
        }

        div[data-testid="stSegmentedControl"] button {
            border-color: #ead5df;
            background-color: #fffafd;
            color: #3f2b35;
        }

        .risk-high {
            padding: 1.2rem;
            border: 1px solid #f4b8b2;
            border-radius: 14px;
            background: #fff1f0;
            color: var(--bad);
            font-weight: 800;
            font-size: 1.05rem;
        }

        .risk-low {
            padding: 1.2rem;
            border: 1px solid #b7e4d8;
            border-radius: 14px;
            background: #effaf6;
            color: var(--good);
            font-weight: 800;
            font-size: 1.05rem;
        }

        .info-card {
            min-height: 150px;
            padding: 1.1rem 1.15rem;
            border: 1px solid #e8edf3;
            border-radius: 14px;
            background: #ffffff;
            box-shadow: 0 10px 28px rgba(16, 24, 40, 0.05);
            color: #344054;
            line-height: 1.65;
        }

        .info-card strong {
            display: block;
            color: var(--berry);
            font-size: 0.92rem;
            margin-bottom: 0.45rem;
        }

        .limit-list {
            padding: 1.1rem 1.35rem;
            border: 1px solid #e8edf3;
            border-radius: 14px;
            background: #ffffff;
            box-shadow: 0 10px 28px rgba(16, 24, 40, 0.05);
            color: #344054;
        }

        .limit-list li {
            margin: 0.35rem 0;
        }

        .preview-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin-bottom: 0.8rem;
        }

        .preview-item {
            padding: 0.8rem 0.9rem;
            border: 1px solid #e8edf3;
            border-radius: 12px;
            background: #ffffff;
            box-shadow: 0 8px 20px rgba(16, 24, 40, 0.04);
        }

        .preview-item .label {
            color: var(--muted);
            font-size: 0.75rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.25rem;
        }

        .preview-item .value {
            color: var(--ink);
            font-size: 0.96rem;
            font-weight: 700;
            overflow-wrap: anywhere;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #e8edf3;
            border-radius: 12px;
            overflow: hidden;
        }

        .footer-note {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.55;
        }

        @media (max-width: 900px) {
            .hero {
                grid-template-columns: 1fr;
            }

            .hero h1 {
                font-size: 2.1rem;
            }

            .preview-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 560px) {
            .preview-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def category_picker(label, options, key):
    st.markdown(f'<div class="field-label">{label}</div>', unsafe_allow_html=True)
    value = st.segmented_control(
        label,
        options,
        default=options[0],
        required=True,
        key=key,
        label_visibility="collapsed",
        width="stretch",
    )
    return value or options[0]


st.markdown(
    """
    <div class="hero">
        <div class="hero-copy">
            <div class="eyebrow">Biostat 212B · Clinical ML Demo</div>
            <h1>Breast Cancer Survival Risk Prediction</h1>
            <p>
                An interactive Streamlit app for exploring a neural-network survival risk model
                using clinical, tumor, surgery, receptor-status, and protein-expression variables.
            </p>
        </div>
        <div class="hero-panel">
            <strong>0.40</strong>
            <span>Selected classification threshold for the final neural network model</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="soft-note">
        This app is for educational and exploratory purposes only. It should not be used for
        real clinical decision-making.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)

with st.container(border=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=55)
        gender = category_picker("Gender", ["FEMALE", "MALE"], "gender")
        tumour_stage = category_picker("Tumour Stage", ["I", "II", "III"], "tumour_stage")
        histology = category_picker(
            "Histology",
            [
                "Infiltrating Ductal Carcinoma",
                "Infiltrating Lobular Carcinoma",
                "Mucinous Carcinoma",
            ],
            "histology",
        )

    with col2:
        protein1 = st.number_input("Protein 1", value=0.0)
        protein2 = st.number_input("Protein 2", value=0.0)
        protein3 = st.number_input("Protein 3", value=0.0)
        protein4 = st.number_input("Protein 4", value=0.0)

    with col3:
        er_status = category_picker("ER Status", ["Positive", "Negative"], "er_status")
        pr_status = category_picker("PR Status", ["Positive", "Negative"], "pr_status")
        her2_status = category_picker("HER2 Status", ["Positive", "Negative"], "her2_status")
        surgery_type = category_picker(
            "Surgery Type",
            [
                "Lumpectomy",
                "Simple Mastectomy",
                "Modified Radical Mastectomy",
                "Other",
            ],
            "surgery_type",
        )

# Create input dataframe
input_data = pd.DataFrame(
    {
        "Age": [age],
        "Gender": [gender],
        "Protein1": [protein1],
        "Protein2": [protein2],
        "Protein3": [protein3],
        "Protein4": [protein4],
        "Tumour_Stage": [tumour_stage],
        "Histology": [histology],
        "ER status": [er_status],
        "PR status": [pr_status],
        "HER2 status": [her2_status],
        "Surgery_type": [surgery_type],
    }
)

st.markdown('<div class="section-title">Input Preview</div>', unsafe_allow_html=True)

preview_items = [
    ("Age", age),
    ("Gender", gender),
    ("Tumour Stage", tumour_stage),
    ("Histology", histology),
    ("Protein 1", f"{protein1:.2f}"),
    ("Protein 2", f"{protein2:.2f}"),
    ("Protein 3", f"{protein3:.2f}"),
    ("Protein 4", f"{protein4:.2f}"),
    ("ER Status", er_status),
    ("PR Status", pr_status),
    ("HER2 Status", her2_status),
    ("Surgery Type", surgery_type),
]

preview_html = '<div class="preview-grid">'
for label, value in preview_items:
    preview_html += (
        f'<div class="preview-item">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'</div>'
    )
preview_html += "</div>"

st.markdown(preview_html, unsafe_allow_html=True)

st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
predict_clicked = st.button("Predict Survival Risk", type="primary", use_container_width=True)
st.caption(f"Current classification threshold: {threshold}")

if predict_clicked:
    probability = model.predict_proba(input_data)[:, 1][0]
    prediction = int(probability >= threshold)

    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
    result_col1, result_col2 = st.columns([0.75, 1.25])

    with result_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Predicted Death Risk Probability</div>
                <div class="value">{probability:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with result_col2:
        if prediction == 1:
            st.markdown(
                '<div class="risk-high">Predicted Class: Higher risk of death</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="risk-low">Predicted Class: Lower risk / alive</div>',
                unsafe_allow_html=True,
            )

    st.info(
        f"A predicted probability greater than or equal to {threshold} is classified as "
        "higher risk of death."
    )

st.markdown('<div class="section-title">Final Model Performance</div>', unsafe_allow_html=True)

performance_df = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall / Sensitivity",
            "Specificity",
            "F1 Score",
            "True Negatives",
            "False Positives",
            "False Negatives",
            "True Positives",
        ],
        "Value": [0.790, 0.500, 0.353, 0.906, 0.414, 58, 6, 11, 6],
    }
)

metric_cols = st.columns(5)
summary_metrics = [
    ("Accuracy", "0.790"),
    ("Precision", "0.500"),
    ("Recall", "0.353"),
    ("Specificity", "0.906"),
    ("F1 Score", "0.414"),
]

for col, (label, value) in zip(metric_cols, summary_metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.dataframe(performance_df, use_container_width=True, hide_index=True)

st.markdown('<div class="section-title">Model Interpretation</div>', unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    st.markdown(
        """
        <div class="info-card">
            <strong>Why this model was selected</strong>
            The neural network model with threshold 0.40 was selected because it achieved the
            highest F1 score among all tested model-threshold combinations. It provided the most
            balanced performance between precision, recall, and specificity.
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
        <div class="info-card">
            <strong>What the results suggest</strong>
            The model still missed 11 of the 17 death cases in the test set. This shows that
            breast cancer survival prediction remains challenging with a small and imbalanced
            dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">Limitations</div>', unsafe_allow_html=True)

st.markdown(
    """
    <ul class="limit-list">
        <li>The dataset is small.</li>
        <li>Death is the minority outcome.</li>
        <li>The model was evaluated on only one train-test split.</li>
        <li>Survival time and follow-up duration were not modeled.</li>
        <li>External validation was not performed.</li>
        <li>The app is for educational demonstration only.</li>
    </ul>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="footer-note">
        Visual design update: structured layout, clinical-themed image banner, responsive columns,
        result cards, and clearer model-performance highlights.
    </p>
    """,
    unsafe_allow_html=True,
)
