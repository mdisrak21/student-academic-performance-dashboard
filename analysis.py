import pandas as pd

# Load dataset
df = pd.read_csv("data/StudentsPerformance.csv")

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DESCRIPTIVE STATISTICS =====")
print(df.describe())

print("\n===== AVERAGE SCORES =====")
print("Math:", df["math score"].mean())
print("Reading:", df["reading score"].mean())
print("Writing:", df["writing score"].mean())

print("\n===== TEST PREPARATION ANALYSIS =====")
print(
    df.groupby("test preparation course")[
        ["math score", "reading score", "writing score"]
    ].mean()
)

print("\n===== PARENTAL EDUCATION ANALYSIS =====")
print(
    df.groupby("parental level of education")[
        ["math score", "reading score", "writing score"]
    ].mean()
)