import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    cred = credentials.Certificate('study-buddy-7306c-firebase-adminsdk-fbsvc-c2d71ba03d.json')
    firebase_admin.initialize_app(cred)

print("Connecting to database: 'study-buddy'...")
try:
    db = firestore.client(database_id='study-buddy')
except ValueError:
    print("Could not find 'study-buddy' DB. Trying default...")
    db = firestore.client()


def inspect_database():
    print("\n--- INSPECTION RESULTS ---")

    # 1. List all collections
    collections = db.collections()
    found_collections = list(collections)

    if not found_collections:
        print("[CRITICAL] The database is totally empty. No collections found.")
        print("SOLUTION: Your seed script might have uploaded to '(default)' instead of 'study-buddy'.")
        return

    for coll in found_collections:
        print(f"\n Collection Found: '{coll.id}'")

        # 2. Check the data inside
        docs = list(coll.stream())
        print(f"   - Count: {len(docs)} documents")

        if docs:
            first_doc = docs[0].to_dict()
            keys = list(first_doc.keys())
            print(f"   - Fields in first doc: {keys}")

            # 3. Check for the specific error causing your issue
            if 'word' in first_doc:
                print(f"   -  'word' field exists. Value: {first_doc['word']}")
            else:
                print(f"   - 'word' field is MISSING. The App ignores these records.")

            if coll.id != 'all_quiz_attempts':
                print(f"   - ⚠ NAME MISMATCH: App looks for 'all_quiz_attempts', but found '{coll.id}'.")


if __name__ == "__main__":
    inspect_database()