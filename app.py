from flask import Flask, request, jsonify
from modelTest import create_mock_data, engineer_features, train_model, get_review_words
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import pandas as pd
from flask_cors import CORS
import joblib
app = Flask(__name__)
CORS(app)
db = None
ML_MODEL = None
CURRENT_DATA_DF = pd.DataFrame() # DataFrame to hold data loaded from Firestore

#mock_data = create_mock_data()
#features = engineer_features(mock_data)
#ml_model = joblib.load("model.joblib")
#mock_data = joblib.load("mock_data.joblib")
try:
    # Use the path to your service account key JSON file
    # Get this file from the Firebase Console -> Project settings -> Service accounts
    cred = credentials.Certificate('study-buddy-7306c-firebase-adminsdk-fbsvc-c2d71ba03d.json')
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
     # Set to None if initialization fails
try:
    # Load the pre-trained Random Forest model
    ML_MODEL = joblib.load("model.joblib")
    print("ML Model loaded successfully.")
except Exception as e:
    print(f"Error loading ML Model: {e}. Review functionality may be impaired.")
# Placeholder for your ML model and current data state

#currently gets the words in mock data for revieW
'''
@app.route('/review', methods=['GET'])
def review():
    user_id = request.args.get('user_id', 'User1')
    words = get_review_words(user_id, mock_data, ml_model, num_words = 15)
    return jsonify({"user": user_id, "review_words": words})
'''


def load_data_from_firestore():
    if db is None:
        return pd.DataFrame()

    print("Loading data from Firestore...")

    # 1. FIX: Use the correct collection name from your screenshot
    try:
        attempts_ref = db.collection('all_quiz_attempts')
        docs = attempts_ref.stream()

        data_list = []
        for doc in docs:
            data = doc.to_dict()

            # 2. FIX: Ensure 'word' exists, otherwise skip this bad record
            if 'word' not in data:
                continue

                # Handle timestamp
            if 'timestamp' in data and hasattr(data['timestamp'], 'to_datetime'):
                data['timestamp'] = data['timestamp'].to_datetime()
            elif 'timestamp' not in data:
                data['timestamp'] = datetime.now()

            data_list.append(data)

        df = pd.DataFrame(data_list)

        if not df.empty:
            # FIX: Ensure is_correct is treated as an integer (1 or 0)
            df['is_correct'] = df['is_correct'].astype(int)

            # FIX: Only process difficulty_score if it actually exists in the DB
            if 'difficulty_score' in df.columns:
                df['difficulty_score'] = df['difficulty_score'].fillna(0).astype(int)

            df = df.sort_values(by=["user_id", "word", "timestamp"]).reset_index(drop=True)

        print(f"Loaded {len(df)} records from Firestore.")
        return df

    except Exception as e:
        print(f"Error loading data from Firestore: {e}")
        return pd.DataFrame()

'''
def load_data_from_firestore():
    """Reads all attempt data from the Firestore 'attempts' collection and converts it to a Pandas DataFrame."""
    if db is None:
        return pd.DataFrame()

    print("Loading data from Firestore...")

    # 1. Get all documents from the 'attempts' collection
    try:
        attempts_ref = db.collection('all_quiz_attempts')
        docs = attempts_ref.stream()

        data_list = []
        for doc in docs:
            data = doc.to_dict()

            # 2. Convert Firestore Timestamp object to Python datetime object
            if 'timestamp' in data and hasattr(data['timestamp'], 'to_datetime'):
                data['timestamp'] = data['timestamp'].to_datetime()
            elif 'timestamp' not in data:
                data['timestamp'] = datetime.now()  # Fallback
            data_list.append(data)

        df = pd.DataFrame(data_list)
        if not df.empty:
            # Ensure correct column types
            df['is_correct'] = df['is_correct'].astype(int)
            df['difficulty_score'] = df['difficulty_score'].astype(int)
            df = df.sort_values(by=["user_id", "word", "timestamp"]).reset_index(drop=True)

        print(f"Loaded {len(df)} records from Firestore.")
        return df

    except Exception as e:
        print(f"Error loading data from Firestore: {e}")
        return pd.DataFrame()


# In a full production setup, you would load and train the model on startup:
# CURRENT_DATA_DF = load_data_from_firestore()
# if not CURRENT_DATA_DF.empty:
#    features_df = engineer_features(CURRENT_DATA_DF.copy())
#    ML_MODEL, _ = train_model(features_df)
#    # joblib.dump(ML_MODEL, "model.joblib") # Save the newly trained model
@app.route('/log_attempt', methods=['POST'])
def log_attempt():
    if db is None:
        return jsonify({"message": "Database not connected."}), 500

    try:
        data = request.json
        user_id = data.get('user_id')
        word = data.get('word')
        # is_correct should be an integer 1 (correct) or 0 (incorrect)
        is_correct = int(data.get('is_correct'))
        # You will need to store or pass the word's difficulty score

        # 1. Validate required fields
        if not all([user_id, word, is_correct is not None]):
            return jsonify({"message": "Missing required fields (user_id, word, is_correct)."}), 400

        # 2. Write the new attempt document to the 'attempts' collection
        new_attempt = {
            'user_id': user_id,
            'word': word,
            'timestamp': firestore.SERVER_TIMESTAMP,  # Use server timestamp for accuracy
            'is_correct': is_correct,
        }

        db.collection('attempts').add(new_attempt)

        # NOTE: For simplicity, we are not immediately retraining the model here.
        # In production, retraining or updating the live data must be done periodically.

        return jsonify({"message": f"Attempt for '{word}' logged successfully."}), 201

    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500


# Assuming you have loaded your model and data as global variables:
# ML_MODEL = joblib.load("model.joblib")
# CURRENT_DATA_DF = load_data_from_firestore() # Data loaded at startup
'''
@app.route('/review', methods=['GET'])
def review():
    user_id = request.args.get('user_id', 'User1')

    # To ensure the latest history is used for feature engineering,
    # you might need to reload the data frequently (or use a sophisticated caching mechanism).
    global CURRENT_DATA_DF
    CURRENT_DATA_DF = load_data_from_firestore()  # Re-load data for review

    if CURRENT_DATA_DF.empty:
        return jsonify({"user": user_id, "review_words": [], "message": "No data found to generate review list."})

    # 1. Re-engineer features on the latest data
    features_df = engineer_features(CURRENT_DATA_DF.copy())

    # 2. Get the review words using the re-engineered features
    try:
        words = get_review_words(user_id, CURRENT_DATA_DF, ML_MODEL, num_words=15)
        return jsonify({"user": user_id, "review_words": words})
    except Exception as e:
        return jsonify({"error": f"Error generating review list: {str(e)}"}), 500
if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 8080, debug=True)
