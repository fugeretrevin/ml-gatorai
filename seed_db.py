import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
# Import your existing data generator
from modelTest import create_mock_data

# --- 1. SETUP FIREBASE ---
# Update this filename to match your NEW json key

if not firebase_admin._apps:
    # Only initialize if no app exists
    cred = credentials.Certificate("study-buddy-7306c-firebase-adminsdk-fbsvc-c2d71ba03d.json")
    firebase_admin.initialize_app(cred)

db = firestore.client(database_id='study-buddy')

def upload_mock_data():
    # --- 2. GENERATE DATA ---
    print("Generating mock data...")
    # You can adjust these numbers to control how much data you upload
    df = create_mock_data(num_users=5, words_per_user=10, attempts_per_word=5)

    print(f"Generated {len(df)} rows. Preparing to upload to 'all_quiz_attempts'...")

    # --- 3. UPLOAD IN BATCHES ---
    collection_name = 'all_quiz_attempts'  # Make sure this matches your DB exactly
    batch = db.batch()
    count = 0
    total_uploaded = 0

    # Convert DataFrame to a list of dictionaries for easy iteration
    records = df.to_dict(orient='records')

    for record in records:
        # DATA CLEANUP: Firestore doesn't like Pandas Timestamps or numpy bools
        # Convert Timestamp to Python datetime
        if isinstance(record.get('timestamp'), pd.Timestamp):
            record['timestamp'] = record['timestamp'].to_pydatetime()

        # Convert numpy bools/ints to native Python types
        record['is_correct'] = bool(record['is_correct'])
        # If you have difficulty_score, ensure it's an int
        if 'difficulty_score' in record:
            record['difficulty_score'] = int(record['difficulty_score'])

        # Create a reference for a new document
        doc_ref = db.collection(collection_name).document()

        # Add the write operation to the batch
        batch.set(doc_ref, record)
        count += 1

        # Firestore allows max 500 writes per batch. We commit every 400 to be safe.
        if count >= 400:
            batch.commit()
            total_uploaded += count
            print(f"Committed {count} records...")
            batch = db.batch()  # Start a new batch
            count = 0

    # Commit any remaining records
    if count > 0:
        batch.commit()
        total_uploaded += count
        print(f"Committed final {count} records.")

    print(f"SUCCESS: Uploaded {total_uploaded} total attempts to Firestore.")


if __name__ == "__main__":
    upload_mock_data()