import numpy as np
from datetime import datetime
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



#MOCK SETUPS FOR DATA, WILL BE INTEGRATED TO DB LATER

#general user data
users = {
    "User1": {
        "words_reviewed": 100,    #total reviewed
        "average_accuracy": 82,   #percent
        "streak_length": 5,       #streak length in days
        "review_frequency": 3.2   #average # words reviewed a day

    }
}

#general word data
words = {           #https://datayze.com/word-analyzer?word=amenable can be used to find difficulties and other data
    "amenable": {
        "word_length": 8,
        "syllable_count": 4,
        "difficulty_score": 5,   #score out of 10
        "part_of_speech": "adjective",

    }
}

interactions = {
    ("User1", "amenable"): {
        "times_seen": 4,
        "times_correct": 3,
        "last_seen": datetime.now() - timedelta(days=3),  #day last seen
        "time_since_last_seen": 3,        #days ago last seen
        "avg_response_time": 10.4,        #average seconds taken per response
        "confidence_rating": 7.5       #calculated value, could be taken by correct / seen * factor of time spent?
    }
}

attempts = [
    {"user_id": "User1", "word_id": "amenable", "time_since_seen": -1, "correct": 0, "acc": 0},
    {"user_id": "User1", "word_id": "amenable", "time_since_seen": 5, "correct": 1, "acc": 0.5},
    {"user_id": "User1", "word_id": "amenable", "time_since_seen": 3, "correct": 1, "acc": 0.66},
    {"user_id": "User1", "word_id": "amenable", "time_since_seen": 2, "correct": 1, "acc": 0.75},
    {"user_id": "User2", "word_id": "amenable", "time_since_seen": 0, "correct": 0, },
    {"user_id": "User2", "word_id": "amenable", "time_since_seen": 5, "correct": 1},
    {"user_id": "User2", "word_id": "amenable", "time_since_seen": 3, "correct": 1},
    {"user_id": "User2", "word_id": "amenable", "time_since_seen": 2, "correct": 1},

]
df = pd.DataFrame(attempts)
accuracy_rate = df["correct"].mean()
avg_time_since_seen = df["time_since_seen"].mean()
accuracy_trend = df["correct"].diff().mean()


#get calculated features for predicting correctness
def get_attributes(user_id, word_id):
    user = users.get(user_id, {})
    word = words.get(word_id, {})
    interaction = interactions.get((user_id, word_id), {})

    times_seen = interaction.get("times_seen", 1)
    times_correct = interaction.get("times_correct", 0)
    accuracy_rate = float(times_correct) / float(times_seen)

    last_seen = interaction.get("time_since_last_seen", datetime.now())
    if not last_seen:  #if hasn't been seen, recency doesn't affect calculation
        recency_weight = 1.0
    else:
        days_since = interaction.get("time_since_last_seen")
        recency_weight = np.exp(-days_since / 7)    #exponsntial decay over the course of a week
        print("recency weight:", recency_weight)
    difficulty = word.get("difficulty_score", 0)

    #get accuracy, recency calculation and difficulty
    attributes = {
        "accuracy_rate": accuracy_rate,
        "recency_weight": recency_weight,
        "difficulty_score": difficulty,
    }
    return attributes

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

#PLACEHOLDER CALCULATION WITHOUT TRAINED ML PROCESS
def predict_correctness(user_id, word_id):
    attributes = get_attributes(user_id, word_id)
    #score calculated mostly by previous accuracy, then last seen, then if it is a difficult word, confidence is reduced by a small amount
    print(attributes)

    X_prediction = pd.DataFrame([attributes])
    probability = model.predict_proba(X_prediction)[0][1]
    return "{:.2f}".format(probability)
    #score = 0.6 * attributes["accuracy_rate"] + 0.3 * attributes["recency_weight"] - 0.02 * attributes["difficulty_score"]

    #return confidence of correct answer
    ###
    #if score > 1:
     #   return 1.0
   # elif score < 0:
    #    return 0.0
   # else:
   #     formatted_score = "{:.2f}".format(score)
    #    return formatted_score
    ###


if __name__ == "__main__":
    user_id = "User1"
    word_id = "amenable"
    prob_correct = predict_correctness(user_id, word_id)
    print("probability correct:", prob_correct)