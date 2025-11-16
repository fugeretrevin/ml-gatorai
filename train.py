import joblib
from modelTest import create_mock_data, engineer_features, train_model

print("Creating mock data...")
mock_data = create_mock_data()

print("Engineering features...")
features = engineer_features(mock_data)

print("Training model...")
ml_model, full_features = train_model(features)

joblib.dump(ml_model, 'model.joblib')
joblib.dump(mock_data, 'mock_data.joblib')

print("Model and mock data saved successfully!")