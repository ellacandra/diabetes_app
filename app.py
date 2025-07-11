import os
import csv
import joblib
import openai
import requests
import pandas as pd
from io import BytesIO
from datetime import datetime
from flask import Flask, render_template, request, redirect, jsonify, make_response
from dotenv import load_dotenv
from xhtml2pdf import pisa

load_dotenv()

app = Flask(__name__)

# === DOWNLOAD MODEL ===
MODEL_URL = "https://drive.google.com/uc?export=download&id=1iYiJHGLujp-hicMoItU4FsTgyscHnTZB"
MODEL_PATH = "model_rf_optuna.pkl"

if not os.path.exists(MODEL_PATH):
    print("🔄 Mengunduh model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model berhasil diunduh.")

model = joblib.load(MODEL_PATH)
CSV_FILE = "riwayat.csv"

# === HALAMAN FORM PREDIKSI ===
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            data = {
                "gender": int(request.form["gender"]),
                "age": int(request.form["age"]),
                "hypertension": int(request.form["hypertension"]),
                "heart_disease": int(request.form["heart_disease"]),
                "smoking_history": int(request.form["smoking_history"]),
                "bmi": float(request.form["bmi"]),
                "hba1c": float(request.form["hba1c"]),
                "blood_glucose_level": float(request.form["blood_glucose_level"])
            }

            df = pd.DataFrame([data])
            prediction = model.predict(df)[0]
            probability = round(model.predict_proba(df)[0][1] * 100, 2)

            tips = ""
            if prediction == 1:
                tips = "Jaga pola makan, rutin cek gula darah, dan lakukan aktivitas fisik secara teratur."
            else:
                tips = "Pertahankan gaya hidup sehat untuk mencegah risiko diabetes."

            row = [*data.values(), prediction, probability, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            with open(CSV_FILE, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row)

            return render_template("result.html", prediction="Positif" if prediction else "Negatif",
                                   probability=probability, tips=tips)

        except Exception as e:
            return f"Terjadi kesalahan: {e}"

    return render_template("index.html")

# === HASIL KE PDF ===
@app.route("/download-pdf", methods=["POST"])
def download_pdf():
    prediction = request.form["prediction"]
    probability = request.form["probability"]
    tips = request.form["tips"]

    rendered = render_template("pdf_template.html", prediction=prediction, probability=probability, tips=tips)
    pdf = BytesIO()
    pisa.CreatePDF(BytesIO(rendered.encode("utf-8")), dest=pdf)

    response = make_response(pdf.getvalue())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=hasil_prediksi.pdf"
    return response

# === RIWAYAT PREDIKSI ===
@app.route("/history")
def history():
    if not os.path.exists(CSV_FILE):
        return render_template("history.html", rows=[])

    df = pd.read_csv(CSV_FILE, header=None)
    df.columns = ["Gender", "Usia", "Hipertensi", "Jantung", "Merokok", "BMI", "HbA1c", "Glukosa", "Prediksi", "Probabilitas", "Waktu"]
    rows = df.to_dict(orient="records")
    return render_template("history.html", rows=rows)

@app.route("/export-history")
def export_history():
    df = pd.read_csv(CSV_FILE, header=None)
    df.columns = ["Gender", "Usia", "Hipertensi", "Jantung", "Merokok", "BMI", "HbA1c", "Glukosa", "Prediksi", "Probabilitas", "Waktu"]
    df.to_excel("riwayat_export.xlsx", index=False)
    return redirect("/history")

@app.route("/delete-history", methods=["POST"])
def delete_history():
    open(CSV_FILE, "w").close()
    return redirect("/history")

# === CHATBOT ===
@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_msg = request.json.get("message")
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Kamu adalah asisten kesehatan yang membantu menjawab pertanyaan seputar diabetes."},
                {"role": "user", "content": user_msg}
            ]
        )
        return jsonify({"response": response.choices[0].message.content.strip()})
    except Exception as e:
        return jsonify({"response": "⚠️ Maaf, terjadi kesalahan saat menghubungi AI."})

if __name__ == "__main__":
    app.run(debug=True)
