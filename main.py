from flask import Flask, request, jsonify
from modelTest import create_mock_data, engineer_features, train_model, get_review_words
import pandas as pd

app = Flask(__name__)

mock_data = create_mock_data()
features = engineer_features(mock_data)
ml_model, full_features = train_model(features)

@app.route('/review', methods=['GET'])
def review():
    user_id = request.args.get('user_id', 'User1')
    words = get_review_words(user_id, mock_data, ml_model, num_words = 15)
    return jsonify({"user": user_id, "review_words": words})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 8080, debug=True)
