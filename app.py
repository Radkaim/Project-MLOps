from flask import Flask, request, jsonify, render_template
import mlflow
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

# --- Cấu hình MLflow ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # MLflow Tracking Server

model_name = "BestClassifier"

# 🔄 Load model bản mới nhất (Production stage hoặc Latest)
print("🔄 Đang load model từ MLflow Registry...")
# model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Production")
model = mlflow.pyfunc.load_model("models:/BestClassifier@Production")

print("✅ Model loaded thành công!")

# 📋 Lấy thông tin version của model
client = mlflow.tracking.MlflowClient()
model_versions = client.search_model_versions(f"name='{model_name}'")
latest_version = max(model_versions, key=lambda x: x.version)
print(f"📋 Model Version: {latest_version.version}")
print(f"📋 Model Stage: {latest_version.current_stage}")
print(f"📋 Model Run ID: {latest_version.run_id}")


# --- Route chính ---
@app.route("/")
def home():
    return render_template("index.html")


# --- API dự đoán ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data]).astype(float)

        prediction = model.predict(df)

        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
