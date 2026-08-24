import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/StudentsPerformance.csv")

# -------------------------------
# 1. Average score by subject
# -------------------------------

average_scores = {
    "Math": df["math score"].mean(),
    "Reading": df["reading score"].mean(),
    "Writing": df["writing score"].mean()
}

plt.figure(figsize=(8, 5))
plt.bar(average_scores.keys(), average_scores.values())

plt.title("Average Student Scores by Subject")
plt.xlabel("Subject")
plt.ylabel("Average Score")
plt.ylim(0, 100)

plt.tight_layout()
plt.savefig("average_scores.png")
plt.show()


# -------------------------------
# 2. Test preparation comparison
# -------------------------------

prep_scores = df.groupby("test preparation course")[
    ["math score", "reading score", "writing score"]
].mean()

prep_scores.plot(kind="bar", figsize=(9, 5))

plt.title("Performance by Test Preparation Course")
plt.xlabel("Test Preparation Course")
plt.ylabel("Average Score")
plt.ylim(0, 100)
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("test_preparation.png")
plt.show()


# -------------------------------
# 3. Score distribution
# -------------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["math score"], bins=10, alpha=0.6, label="Math")
plt.hist(df["reading score"], bins=10, alpha=0.6, label="Reading")
plt.hist(df["writing score"], bins=10, alpha=0.6, label="Writing")

plt.title("Distribution of Student Scores")
plt.xlabel("Score")
plt.ylabel("Number of Students")
plt.legend()

plt.tight_layout()
plt.savefig("score_distribution.png")
plt.show()