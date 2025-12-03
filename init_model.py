import joblib
from modelTest import create_mock_data, engineer_features, train_model

print("Creating mock data...")
mock_data = create_mock_data()

print("Engineering features...")
features = engineer_features(mock_data)

print("Training model...")
model, _ = train_model(features)

print("Saving model to model.joblib...")
joblib.dump(model, "model.joblib")

joblib.dump(mock_data, "mock_data.joblib")

print("Done, run app.py")