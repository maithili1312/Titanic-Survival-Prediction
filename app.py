from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model (full pipeline)
model_path='model.pkl'
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Map API input -> model features
    input_dict = {
        "pclass": data["pclass"],
        "sex": data["sex"],
        "age": data["age"],
        "sibsp": data["sibsp"],
        "parch": data["parch"],
        "fare": data["fare"],
        "class": data["class_"],   # 🔥 important mapping
        "who": data["who"],
        "adult_male": bool(data["adult_male"]),
        "alone": bool(data["alone"])
    }

    feature_order = [
        "pclass", "sex", "age", "sibsp", "parch",
        "fare", "class", "who", "adult_male", "alone"
    ]

    df = pd.DataFrame([input_dict])[feature_order]

    prediction = model.predict(df)

    return jsonify({"survived": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
