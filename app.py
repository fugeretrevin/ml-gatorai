from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import joblib
from datetime import datetime
import os

# Import your helper functions
from modelTest import create_mock_data, engineer_features, train_model, get_review_words

app = Flask(__name__)
CORS(app)

# Global variables
db = None
ML_MODEL = None
CURRENT_DATA_DF = pd.DataFrame()

# --- 1. INITIALIZE FIREBASE ---
try:
    if not firebase_admin._apps:
        # Define the local filename
        key_file = 'study-buddy-7306c-firebase-adminsdk-fbsvc-c2d71ba03d.json'

        if os.path.exists(key_file):
            # We are Local: Use the file
            print(f"Found local key: {key_file}")
            cred = credentials.Certificate(key_file)
            firebase_admin.initialize_app(cred)
        else:
            # We are on Cloud Run: Use Automatic Cloud Credentials
            print("No local key found. Using Cloud Default Credentials.")
            firebase_admin.initialize_app()  # No args needed on Cloud Run!

    # Always connect to the specific DB
    db = firestore.client(database_id='study-buddy')
    print(f"Firebase connected to 'study-buddy'.")

except Exception as e:
    print(f"Error initializing Firebase: {e}")
    db = None
# --- 2. LOAD ML MODEL ---
try:
    ML_MODEL = joblib.load("model.joblib")
    print("ML Model loaded successfully.")
except Exception as e:
    print(f"Error loading ML Model: {e}. Review functionality may be impaired.")


def load_data_from_firestore():
    """Fetches data from 'all_quiz_attempts', cleans it, and returns a DataFrame."""
    if db is None:
        return pd.DataFrame()

    print("Loading data from Firestore...")

    try:
        # TARGET THE CORRECT COLLECTION
        attempts_ref = db.collection('all_quiz_attempts')
        docs = attempts_ref.stream()

        data_list = []
        for doc in docs:
            data = doc.to_dict()

            # Filter: Ensure 'word' exists
            if 'word' not in data:
                continue

            # Basic cleanup (we handle the heavy lifting in Pandas later)
            if 'timestamp' not in data:
                data['timestamp'] = datetime.now()

            data_list.append(data)

        df = pd.DataFrame(data_list)

        if not df.empty:
            # --- FIX: GLOBALLY STRIP TIMEZONE INFO ---
            # 1. Force convert to datetime objects
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # 2. Remove timezone (make it 'naive') so it matches datetime.now()
            # This handles mixed data types safely
            try:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            except Exception:
                # If data is already naive, this might fail, so we ignore
                pass

            # Type conversion
            df['is_correct'] = df['is_correct'].astype(int)

            # Handle optional difficulty score
            if 'difficulty_score' in df.columns:
                df['difficulty_score'] = df['difficulty_score'].fillna(0).astype(int)

            # Sort for feature engineering
            df = df.sort_values(by=["user_id", "word", "timestamp"]).reset_index(drop=True)

        print(f"Loaded {len(df)} records from Firestore.")
        return df

    except Exception as e:
        print(f"Error loading data from Firestore: {e}")
        return pd.DataFrame()
@app.route('/review', methods=['GET'])
@app.route('/review', methods=['GET'])
def review():
    target_user_id = request.args.get('user_id', 'User1')
    print(f"Generating review for: {target_user_id}")

    # 1. Load data
    global CURRENT_DATA_DF
    CURRENT_DATA_DF = load_data_from_firestore()

    if CURRENT_DATA_DF.empty:
        return jsonify({
            "user_id": target_user_id,
            "review_words": [],
            "message": "No data found in Firestore."
        })

    # 2. Engineer features
    try:
        # Create a copy so we can modify it
        working_df = CURRENT_DATA_DF.copy()

        # KEY FIX: engineer_features modifies 'working_df' in-place
        # (adding 'times_seen') even though it returns a cleaned version.
        # We ignore the return value here and use working_df directly.
        engineer_features(working_df)

    except Exception as e:
        return jsonify({"error": f"Feature engineering failed: {str(e)}"}), 500

    # 3. Predict
    try:
        # KEY FIX: Pass 'working_df' (which now has 'times_seen')
        # instead of 'CURRENT_DATA_DF' (which does not)
        words = get_review_words(target_user_id, working_df, ML_MODEL, num_words=15)

        return jsonify({
            "user_id": target_user_id,
            "review_words": words
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
@app.route('/log_attempt', methods=['POST'])
def log_attempt():
    if db is None:
        return jsonify({"message": "Database not connected."}), 500

    try:
        data = request.json
        user_id = data.get('user_id')
        word = data.get('word')
        is_correct = data.get('is_correct')  # Expecting 0 or 1, or boolean

        # Validate inputs
        if not all([user_id, word, is_correct is not None]):
            return jsonify({"message": "Missing required fields."}), 400

        # Create the record
        new_attempt = {
            'user_id': user_id,
            'word': word,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'is_correct': bool(int(is_correct)),  # Ensure stored as boolean or int consistently
        }

        # Write to the same collection read from
        db.collection('all_quiz_attempts').add(new_attempt)

        return jsonify({"message": f"Attempt for '{word}' logged successfully."}), 201

    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)