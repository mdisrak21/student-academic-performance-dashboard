import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. Load dataset
# ==========================================

df = pd.read_csv("data/StudentsPerformance.csv")


# ==========================================
# 2. Select features and target
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


# ==========================================
# 3. Define categorical and numerical columns
# ==========================================

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


# ==========================================
# 4. Preprocessing
# ==========================================

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


# ==========================================
# 5. Create Machine Learning pipeline
# ==========================================

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


# ==========================================
# 6. Split data
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# ==========================================
# 7. Train model
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 8. Make predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 9. Evaluate model
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")


# ==========================================
# 10. Example prediction
# ==========================================

example_student = pd.DataFrame(
    [
        {
            "gender": "female",
            "race/ethnicity": "group C",
            "parental level of education": "bachelor's degree",
            "lunch": "standard",
            "test preparation course": "completed",
            "reading score": 80,
            "writing score": 78,
        }
    ]
)

prediction = model.predict(example_student)

print("\n===== EXAMPLE PREDICTION =====")
print(f"Predicted Math Score: {prediction[0]:.2f}")