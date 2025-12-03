import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc


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

        user_words = np.random.choice(words, min(len(words), 10), replace=False)

        for word in user_words:
            is_correct_history = []
            for j in range(attempts_per_word):
                # Logic: User gets better over time (prob increases)
                if len(is_correct_history) == 0:
                    prob_correct = 0.3
                else:
                    prob_correct = sum(is_correct_history) / len(is_correct_history)
                    prob_correct = min(0.9, prob_correct + 0.1)  # Cap at 90%

                correct = 1 if np.random.rand() < prob_correct else 0
                is_correct_history.append(correct)

                # Random timestamp in the past
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
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['is_correct'] = df['is_correct'].astype(bool)

    print(f"Generated {len(df)} rows of mock data.")
    return df




def fetch_data_from_firestore():
    print("Fetching data from Firestore...")
    attempts_ref = db.collection("all_quiz_attempts")
    docs = attempts_ref.stream()
    attempts_list = []
    for doc in docs:
        attempts_list.append(doc.to_dict())
    
    if not attempts_list:
        print("No data found in Firestore.")
        return pd.DataFrame()

    df = pd.DataFrame(attempts_list)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['is_correct'] = df['is_correct'].astype(bool)

    print(f"Successfully fetched {len(df)} attempts.")
    return df


def engineer_features(df):
    # time since the last attempt for user, work
    df['is_correct_int'] = df['is_correct'].astype(int)
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


def plot_classification_report(y_test, y_pred):
    from sklearn.metrics import classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    report_dict = classification_report(y_test, y_pred, output_dict=True)

    report_df = pd.DataFrame(report_dict).transpose()

    plot_df = report_df.drop(columns=['support'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(plot_df, annot=True, cmap='RdBu_r', vmin=0, vmax=1.0, fmt='.2%')
    plt.title('Model Classification Report')
    plt.show()


def visualize_model_performance(model, X_test, y_test, feature_names):
    sns.set_theme(style="whitegrid")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Weights)")
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette="viridis")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.show()

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Incorrect', 'Correct'],
                yticklabels=['Incorrect', 'Correct'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    plot_df = report_df.drop(columns=['support'])  # drop support col for cleaner map

    plt.figure(figsize=(8, 6))
    sns.heatmap(plot_df, annot=True, cmap='RdBu_r', vmin=0, vmax=1.0, fmt='.2%')
    plt.title('Model Classification Report')
    plt.show()
# train model based on features
def train_and_evaluate(features_df):
    features = [
        'time_since_last_seen_days',
        'prev_attempt_correct',
        'times_seen',
        'word_accuracy_rate',
        'user_overall_accuracy'
    ]
    target = 'is_correct'

    X = features_df[features]
    y = features_df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    visualize_model_performance(model, X_test, y_test, features)

    return model
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

    # NEW LINE: Returns a list of dictionaries like [{'word': 'apple', 'prob': 0.45}, ...]
    return review_list[['word', 'predicted_prob_correct']].head(num_words).to_dict(orient='records')



if __name__ == "__main__":
    data_df = create_mock_data(num_users=10, words_per_user=15, attempts_per_word=10)
    '''
    if not data_df.empty:
        # 2. Engineer features
        features_df = engineer_features(data_df)

        # 3. Train and Visualize
        ml_model = train_and_evaluate(features_df)
        # 4. Test Prediction Logic (Optional)
        # mock_user = "User1"
        # get_review_words(mock_user, data_df, ml_model)
    '''
    real_data_df = fetch_data_from_firestore()
    if not real_data_df.empty:
        # get features from mock data
        features_df = engineer_features(real_data_df)

        # train model
        ml_model, full_feature_df = train_model(features_df)

        mock_user = "User1"
        review_session_words = get_review_words(mock_user, real_data_df, ml_model, num_words=15)
