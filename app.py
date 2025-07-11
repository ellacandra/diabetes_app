import openai
from flask import Flask, render_template, request, redirect, make_response, jsonify
import joblib, csv, os
from datetime import datetime
from xhtml2pdf import pisa
from io import BytesIO
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
model = joblib.load("model_rf_optuna.pkl")
CSV_FILE = "riwayat.csv"

# 🔑

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        smoking_history = int(request.form['smoking_history'])
        bmi = float(request.form['bmi'])
        hba1c = float(request.form['hba1c'])
        glucose = float(request.form['blood_glucose_level'])

        features = [[gender, age, hypertension, heart_disease,
                     smoking_history, bmi, hba1c, glucose]]
        prob = model.predict_proba(features)[0][1]
        prediction = "Diabetes" if prob >= 0.5 else "No Diabetes"
        probability = round(prob * 100, 2)

        tips = "✅ Tidak terindikasi diabetes. Tetap jaga pola hidup sehat."
        if prediction == "Diabetes":
            tips = "⚠️ Hasil menunjukkan kemungkinan diabetes. Konsultasikan ke dokter dan jaga pola makan."

        if bmi > 30:
            tips += " Berat badan masuk kategori obesitas."
        elif bmi < 18.5:
            tips += " Berat badan di bawah normal."

        if hba1c > 6.5:
            tips += " HbA1c tergolong tinggi."
        elif hba1c > 5.7:
            tips += " HbA1c dalam rentang pre-diabetes."

        if glucose > 200:
            tips += " Kadar glukosa sangat tinggi."

        # Simpan ke CSV
        if not os.path.isfile(CSV_FILE):
            with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'gender', 'age', 'hypertension', 'heart_disease',
                                 'smoking_history', 'bmi', 'hba1c', 'glucose', 'prediction', 'probability'])

        with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             gender, age, hypertension, heart_disease,
                             smoking_history, bmi, hba1c, glucose, prediction, probability])

        return render_template('result.html', prediction=prediction, probability=probability, tips=tips)

    return render_template('index.html')

@app.route('/history')
def history():
    filter_val = request.args.get('filter')
    records = []
    if os.path.isfile(CSV_FILE):
        with open(CSV_FILE, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            records = list(reader)
        if filter_val:
            records = [r for r in records if r['prediction'] == filter_val]
    return render_template('history.html', records=records)

@app.route('/clear-history', methods=['POST'])
def clear_history():
    if os.path.isfile(CSV_FILE):
        os.remove(CSV_FILE)
    return redirect('/history')

@app.route('/export-excel')
def export_excel():
    if os.path.isfile(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        file_path = "riwayat_export.xlsx"
        df.to_excel(file_path, index=False)
        with open(file_path, "rb") as f:
            response = make_response(f.read())
            response.headers["Content-Disposition"] = "attachment; filename=riwayat_prediksi.xlsx"
            response.headers["Content-Type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            return response
    return "Tidak ada data riwayat untuk diekspor.", 404

@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    prediction = request.form.get('prediction', 'Tidak tersedia')
    probability = request.form.get('probability', '0')
    tips = request.form.get('tips', 'Tidak ada tips tersedia')

    html = render_template('pdf_template.html', prediction=prediction, probability=probability, tips=tips)
    result = BytesIO()
    pdf = pisa.pisaDocument(BytesIO(html.encode("UTF-8")), result)

    if not pdf.err:
        response = make_response(result.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=hasil_prediksi.pdf'
        return response
    else:
        return "Gagal membuat PDF", 500

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Kamu adalah asisten kesehatan yang ramah dan ahli dalam menjelaskan tentang diabetes."},
                {"role": "user", "content": user_msg}
            ]
        )
        reply = response.choices[0].message["content"]
        return jsonify({"reply": reply})
    except Exception as e:
        print("Chat Error:", e)
        return jsonify({"reply": "⚠️ Maaf, terjadi kesalahan saat menghubungi AI."}), 500


if __name__ == '__main__':
    app.run(debug=True)
