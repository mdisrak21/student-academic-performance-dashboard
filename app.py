import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Student Academic Performance",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# Load Dataset
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("data/StudentsPerformance.csv")


df = load_data()


# ==========================================
# Train Model
# ==========================================

X = df[
    [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
        "reading score",
        "writing score",
    ]
]

y = df["math score"]


categorical_features = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]

numerical_features = [
    "reading score",
    "writing score",
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "numerical",
            "passthrough",
            numerical_features,
        ),
    ]
)


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
            ),
        ),
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model.fit(X_train, y_train)


# ==========================================
# Header
# ==========================================

st.title("🎓 Student Academic Performance Dashboard")

st.write(
    "An interactive dashboard for exploring student performance "
    "and predicting mathematics scores using machine learning."
)


# ==========================================
# Dataset Overview
# ==========================================

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Students", len(df))

with col2:
    st.metric(
        "Average Math Score",
        f"{df['math score'].mean():.2f}"
    )

with col3:
    st.metric(
        "Average Reading Score",
        f"{df['reading score'].mean():.2f}"
    )


# ==========================================
# Subject Performance
# ==========================================

st.header("📈 Average Subject Scores")

average_scores = pd.DataFrame(
    {
        "Subject": ["Math", "Reading", "Writing"],
        "Average Score": [
            df["math score"].mean(),
            df["reading score"].mean(),
            df["writing score"].mean(),
        ],
    }
)

st.bar_chart(
    average_scores.set_index("Subject")
)


# ==========================================
# Test Preparation Analysis
# ==========================================

st.header("📝 Test Preparation Analysis")

prep_scores = df.groupby("test preparation course")[
    ["math score", "reading score", "writing score"]
].mean()

st.dataframe(
    prep_scores.round(2),
    use_container_width=True
)


# ==========================================
# Student Prediction
# ==========================================

st.header("🤖 Predict Student Math Score")

st.write(
    "Enter student information below to generate an estimated "
    "mathematics score."
)


col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        sorted(df["gender"].unique())
    )

    race = st.selectbox(
        "Race/Ethnicity",
        sorted(df["race/ethnicity"].unique())
    )

    parental_education = st.selectbox(
        "Parental Level of Education",
        sorted(df["parental level of education"].unique())
    )

    lunch = st.selectbox(
        "Lunch",
        sorted(df["lunch"].unique())
    )


with col2:

    preparation = st.selectbox(
        "Test Preparation Course",
        sorted(df["test preparation course"].unique())
    )

    reading_score = st.slider(
        "Reading Score",
        min_value=0,
        max_value=100,
        value=70
    )

    writing_score = st.slider(
        "Writing Score",
        min_value=0,
        max_value=100,
        value=70
    )


# ==========================================
# Prediction
# ==========================================

if st.button("🔮 Predict Math Score"):

    student = pd.DataFrame(
        [
            {
                "gender": gender,
                "race/ethnicity": race,
                "parental level of education": parental_education,
                "lunch": lunch,
                "test preparation course": preparation,
                "reading score": reading_score,
                "writing score": writing_score,
            }
        ]
    )

    prediction = model.predict(student)[0]

    st.success(
        f"Predicted Math Score: {prediction:.2f}"
    )


# ==========================================
# Data Explorer
# ==========================================

st.header("🔍 Explore Dataset")

st.dataframe(
    df,
    use_container_width=True
)


# ==========================================
# Download Dataset
# ==========================================

st.download_button(
    label="📥 Download Dataset",
    data=df.to_csv(index=False),
    file_name="StudentsPerformance.csv",
    mime="text/csv"
)


# ==========================================
# Footer
# ==========================================

st.markdown("---")

st.caption(
    "Developed by Md. Israk Ahmmed | "
    "Computer Science & Engineering | University of Barishal"
)