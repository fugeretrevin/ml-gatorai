import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# random generation, can be used to see visual correlations

np.random.seed(42)
num_samples = 100
train_data = pd.DataFrame ({
    "accuracy_rate": np.random.uniform(0, 1, num_samples),
    "recency_weight": np.random.uniform(0.1, 1, num_samples),
    "difficulty_score": np.random.uniform(1, 10, num_samples),
})

train_data["is_correct"] = (
(train_data["accuracy_rate"] * 0.6 + train_data["recency_weight"] * 0.3 - train_data["difficulty_score"] * 0.02)
    + np.random.normal(0, 0.1, num_samples)) > 0.5
train_data["is_correct"] = train_data["is_correct"].astype(int)


X = train_data[["accuracy_rate", "recency_weight", "difficulty_score"]]
Y = train_data["is_correct"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)
plt.plot(X, Y)
plt.xlabel("acc diff tsc")
plt.ylabel("correct")
plt.show()
probability = model.predict_proba(X_test)
print(probability)

def predict_probability(accuracy, difficulty, time_since_seen):
    return model.predict_proba([[accuracy, difficulty, time_since_seen]])[0][1]