from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ✅ Load trained XGBoost model & encoders
with open("model/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Get input values
        gender = request.form.get('gender', 'Unknown')
        part_time_job = request.form.get('part_time_job', 'No')
        absence_days = float(request.form.get('absence_days', 0))
        extracurricular = request.form.get('extracurricular', 'No')
        self_study = float(request.form.get('self_study', 0))
        career_aspiration = request.form.get('career_aspiration', 'None')
        math_score = float(request.form.get('math_score', 0))
        physics_score = float(request.form.get('physics_score', 0))
        chemistry_score = float(request.form.get('chemistry_score', 0))
        biology_score = float(request.form.get('biology_score', 0))
        english_score = float(request.form.get('english_score', 0))

        # ✅ Encoding function
        def safe_encode(value, column_name):
            if column_name in label_encoders:
                if value in label_encoders[column_name].classes_:
                    return label_encoders[column_name].transform([value])[0]
                else:
                    return 0  
            return 0  

        gender = safe_encode(gender, 'gender')
        part_time_job = safe_encode(part_time_job, 'part_time_job')
        extracurricular = safe_encode(extracurricular, 'extracurricular_activities')
        career_aspiration = safe_encode(career_aspiration, 'career_aspiration')

        # ✅ Prepare input array
        features = np.array([[gender, part_time_job, absence_days, extracurricular, self_study, career_aspiration,
                              math_score, physics_score, chemistry_score, biology_score, english_score]])

        # ✅ Predict
        prediction = model.predict(features)[0]

        return render_template('result.html', prediction=round(prediction, 2))

    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
