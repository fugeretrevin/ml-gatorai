import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# model most useful for full integration

def create_mock_data(num_users=5, words_per_user=20, attempts_per_word=5):
    words = [
        "amenable", "ephemeral", "gregarious", "juxtaposition", "ubiquitous",
        "pulchritudinous", "sycophant", "anachronistic", "laconic", "magnanimous",
        "obfuscate", "plethora", "quixotic", "recalcitrant", "veracity",
        "zephyr", "cognizant", "ebullient", "fastidious", "idiosyncratic"
    ]

    attempts_data = []

    for i in range(1, num_users + 1):
        #create users
        user_id = f"User{i}"
        for word in np.random.choice(words, words_per_user, replace=False):
            # fake history for attempt datas
            is_correct_history = []
            for j in range(attempts_per_word):
                #general trend of more often correct if seen more and gotten right
                prob_correct = sum(is_correct_history) / (len(is_correct_history) + 2) + 0.2
                correct = 1 if np.random.rand() < prob_correct else 0
                is_correct_history.append(correct)

                attempt_timestamp = datetime.now() - timedelta(days=np.random.randint(1, 100),
                                                               hours=np.random.randint(0, 24))

                attempts_data.append({
                    "user_id": user_id,
                    "word": word,
                    "timestamp": attempt_timestamp,
                    "is_correct": correct,
                })

    df = pd.DataFrame(attempts_data)
    df = df.sort_values(by=["user_id", "word", "timestamp"]).reset_index(drop=True)
    print(df)
    return df


cred = credentials.Certificate("C:/Users/12148/Documents/GitHub/ml-gatorai/study-buddy-7306c-firebase-adminsdk-fbsvc-c98a89a3c4.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# --- 2. NEW: Function to fetch all data ---
def fetch_data_from_firestore():
    print("Fetching data from Firestore...")
    attempts_ref = db.collection("all_quiz_attempts")
    docs = attempts_ref.stream() # This gets all documents in the collection

    attempts_list = []
    for doc in docs:
        attempts_list.append(doc.to_dict())
    
    if not attempts_list:
        print("No data found in Firestore.")
        return pd.DataFrame() # Return empty DataFrame

    # --- 3. Convert to the DataFrame your model expects ---
    df = pd.DataFrame(attempts_list)
    
    # IMPORTANT: Convert data types
    # Timestamps from Firestore are already datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['is_correct'] = df['is_correct'].astype(bool)
    # The rest (user_id, word) are probably fine as objects/strings
    
    print(f"Successfully fetched {len(df)} attempts.")
    return df


def engineer_features(df):
    # time since the last attempt for user, work
    df['time_since_last_seen_days'] = df.groupby(['user_id', 'word'])['timestamp'].diff().dt.total_seconds() / (
                60 * 60 * 24)
    df['time_since_last_seen_days'] = df['time_since_last_seen_days'].fillna(
        0)  # first attempt is 0 days since last seen

    #  if last attempt was correct
    df['prev_attempt_correct'] = df.groupby(['user_id', 'word'])['is_correct'].shift(1).fillna(0)


    df['times_seen'] = df.groupby(['user_id', 'word']).cumcount() + 1
    df['times_correct'] = df.groupby(['user_id', 'word'])['is_correct'].cumsum()

    # calculates word accuracy before curr attempt
    df['word_accuracy_rate'] = (df.groupby(['user_id', 'word'])['is_correct'].shift(1).fillna(0).cumsum()) / df[
        'times_seen']

    # total performance
    df['user_total_attempts'] = df.groupby('user_id').cumcount() + 1
    df['user_correct_attempts'] = df.groupby('user_id')['is_correct'].cumsum()
    df['user_overall_accuracy'] = (df.groupby('user_id')['is_correct'].shift(1).fillna(0).cumsum()) / df[
        'user_total_attempts']

    # remove unneeded columns
    features_df = df.drop(columns=['timestamp', 'user_total_attempts', 'user_correct_attempts'])

    # fill missing data
    features_df = features_df.fillna(0)

    return features_df


# train model based on features
def train_model(features_df):

    features = [
        'time_since_last_seen_days',
        'prev_attempt_correct',
        'times_seen',
        'word_accuracy_rate',
        'user_overall_accuracy'
    ]
    target = 'is_correct'

    X = features_df[features]
    y = features_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # initialize and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # model accuracy
    y_pred = model.predict(X_test)
    print("--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("------------------------\n")

    # find importance
    feature_importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    print("--- Feature Importances ---")
    print(feature_importances.sort_values(by='importance', ascending=False))
    print("---------------------------\n")

    return model, features_df



def get_review_words(user_id, all_words_df, model, num_words=15):

    print(f"Review List for {user_id}")

    # get all words for user
    user_latest_state = all_words_df[all_words_df['user_id'] == user_id].groupby('word').last().reset_index()

    if user_latest_state.empty:
        print(f"No learning history found for {user_id}.")
        return []

    # calc 'time_since_last_seen' feature
    now = datetime.now()

    # timestamp to right format
    user_latest_state['timestamp'] = pd.to_datetime(user_latest_state['timestamp'])
    user_latest_state['time_since_last_seen_days'] = ((now - user_latest_state['timestamp']).dt.total_seconds()
    / (60 * 60 * 24))

    # next attempt is one more than last time seen
    user_latest_state['times_seen'] += 1

    user_latest_state['prev_attempt_correct'] = user_latest_state['is_correct']

    # features for prediction
    prediction_features = [
        'time_since_last_seen_days',
        'prev_attempt_correct',
        'times_seen',
        'word_accuracy_rate',
        'user_overall_accuracy'
    ]

    X_predict = user_latest_state[prediction_features]

    # predict probability of getting words correct
    probabilities = model.predict_proba(X_predict)[:, 1]

    user_latest_state['predicted_prob_correct'] = probabilities

    # sort by lowest probability of being correct as those are most likely to be forgotten
    review_list = user_latest_state.sort_values(by='predicted_prob_correct', ascending=True)

    print("Suggested words:")
    print(review_list[['word', 'predicted_prob_correct']].head(num_words))

    return review_list['word'].head(num_words).tolist()


if __name__ == "__main__":
    #mock_data_df = create_mock_data()
    real_data_df = fetch_data_from_firestore()
    if not real_data_df.empty:
        # get features from mock data
        features_df = engineer_features(real_data_df)

        # train model
        ml_model, full_feature_df = train_model(features_df)

        mock_user = "User1"
        review_session_words = get_review_words(mock_user, real_data_df, ml_model, num_words=15)
