import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



def create_mock_data(num_users=5, words_per_user=20, attempts_per_word=5):
    words = [
        "amenable", "ephemeral", "gregarious", "juxtaposition", "ubiquitous",
        "pulchritudinous", "sycophant", "anachronistic", "laconic", "magnanimous",
        "obfuscate", "plethora", "quixotic", "recalcitrant", "veracity",
        "zephyr", "cognizant", "ebullient", "fastidious", "idiosyncratic"
    ]
    word_difficulties = {word: np.random.randint(1, 11) for word in words}

    attempts_data = []

    for i in range(1, num_users + 1):
        #create users
        user_id = f"User{i}"
        for word in np.random.choice(words, words_per_user, replace=False):
            # Simulate a learning history for each word
            is_correct_history = []
            for j in range(attempts_per_word):
                # The more a user sees a word, the more likely they are to be correct
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
                    "difficulty_score": word_difficulties[word]
                })

    df = pd.DataFrame(attempts_data)
    df = df.sort_values(by=["user_id", "word", "timestamp"]).reset_index(drop=True)
    print(df)
    return df


# Create features from  attempt data

def engineer_features(df):
    # Time since the last attempt for the same user-word pair
    df['time_since_last_seen_days'] = df.groupby(['user_id', 'word'])['timestamp'].diff().dt.total_seconds() / (
                60 * 60 * 24)
    df['time_since_last_seen_days'] = df['time_since_last_seen_days'].fillna(
        0)  # First attempt has 0 days since last seen

    # Lag features: performance on the previous attempt
    df['prev_attempt_correct'] = df.groupby(['user_id', 'word'])['is_correct'].shift(1).fillna(0)

    # Cumulative history features for each user-word pair
    df['times_seen'] = df.groupby(['user_id', 'word']).cumcount() + 1
    df['times_correct'] = df.groupby(['user_id', 'word'])['is_correct'].cumsum()

    # Rolling accuracy for the user on that specific word (excluding current attempt)
    # We shift the cumulative sum to calculate accuracy *before* the current attempt
    df['word_accuracy_rate'] = (df.groupby(['user_id', 'word'])['is_correct'].shift(1).fillna(0).cumsum()) / df[
        'times_seen']

    # Overall user performance (calculated up to the point of the attempt)
    df['user_total_attempts'] = df.groupby('user_id').cumcount() + 1
    df['user_correct_attempts'] = df.groupby('user_id')['is_correct'].cumsum()
    df['user_overall_accuracy'] = (df.groupby('user_id')['is_correct'].shift(1).fillna(0).cumsum()) / df[
        'user_total_attempts']

    # Final feature set - drop intermediate columns
    features_df = df.drop(columns=['timestamp', 'user_total_attempts', 'user_correct_attempts'])

    # Handle potential NaNs from division by zero, etc.
    features_df = features_df.fillna(0)

    return features_df


# train model based on features
def train_model(features_df):

    # Define features (X) and target (y)
    features = [
        'difficulty_score',
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

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("------------------------\n")

    # Feature Importance
    feature_importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    print("--- Feature Importances ---")
    print(feature_importances.sort_values(by='importance', ascending=False))
    print("---------------------------\n")

    return model, features_df


# --- 4. Using the Model for Predictions ---

def get_review_words(user_id, all_words_df, model, num_words=15):
    """
    Predicts the probability of getting each word correct for a given user
    and returns the words the user is most likely to forget.
    """
    print(f"--- Generating Review List for {user_id} ---")

    # Get the latest state for each word the user has studied
    user_latest_state = all_words_df[all_words_df['user_id'] == user_id].groupby('word').last().reset_index()

    if user_latest_state.empty:
        print(f"No learning history found for {user_id}.")
        return []

    # Calculate the 'time_since_last_seen' feature for today
    now = datetime.now()
    # Ensure 'timestamp' is in datetime format before subtraction
    user_latest_state['timestamp'] = pd.to_datetime(user_latest_state['timestamp'])
    user_latest_state['time_since_last_seen_days'] = (now - user_latest_state['timestamp']).dt.total_seconds() / (
                60 * 60 * 24)

    # The 'next' attempt would be one more than the last 'times_seen'
    user_latest_state['times_seen'] += 1

    # The 'prev_attempt_correct' is the 'is_correct' from the last attempt
    user_latest_state['prev_attempt_correct'] = user_latest_state['is_correct']

    # Define the feature set for prediction
    prediction_features = [
        'difficulty_score',
        'time_since_last_seen_days',
        'prev_attempt_correct',
        'times_seen',
        'word_accuracy_rate',
        'user_overall_accuracy'
    ]

    X_predict = user_latest_state[prediction_features]

    # predict the probability of getting words correct
    probabilities = model.predict_proba(X_predict)[:, 1]

    user_latest_state['predicted_prob_correct'] = probabilities

    # sort by lowest probability of being correct as those are most likely to be forgotten
    review_list = user_latest_state.sort_values(by='predicted_prob_correct', ascending=True)

    print("Suggested words:")
    print(review_list[['word', 'predicted_prob_correct']].head(num_words))

    return review_list['word'].head(num_words).tolist()


if __name__ == "__main__":
    # 1. Create a dataset of historical attempts
    raw_data_df = create_mock_data()

    # 2. Engineer features from this historical data
    features_df = engineer_features(raw_data_df)

    # 3. Train the model on the full historical dataset
    ml_model, full_feature_df = train_model(features_df)

    # 4. Use the trained model to generate a personalized review list for a specific user
    # We pass the original raw data (raw_data_df) to simulate having access to the full DB
    target_user = "User1"
    review_session_words = get_review_words(target_user, raw_data_df, ml_model, num_words=15)
