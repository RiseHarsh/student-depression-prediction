from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load saved model, scaler, encoders, and feature names
model = joblib.load('model/logistic_model.pkl')
scaler = joblib.load('model/scaler.pkl')
label_encoders = joblib.load('model/encoders.pkl')
feature_names = joblib.load('model/feature_names.pkl')  # ✅ NEW

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Get all form inputs
            user_input = {
                'Gender': request.form['gender'],
                'Age': int(request.form['age']),
                'City': request.form['city'],
                'Profession': 'Student',
                'Academic Pressure': float(request.form['academic_pressure']),
                'CGPA': float(request.form['cgpa']),
                'Study Satisfaction': float(request.form['study_satisfaction']),
                'Sleep Duration': float(request.form['sleep_duration']),
                'Dietary Habits': request.form['dietary_habits'],
                'Degree': request.form['degree'],
                'Have you ever had suicidal thoughts ?': request.form['suicidal_thoughts'],
                'Study Hours per Day': float(request.form['study_hours']),
                'Financial Stress': float(request.form['financial_stress']),
                'Family History of Mental Illness': request.form['family_history']
            }

            input_df = pd.DataFrame([user_input])

            # Encode categorical columns
            for col in input_df.select_dtypes(include='object').columns:
                if col in label_encoders:
                    input_df[col] = label_encoders[col].transform(input_df[col])

            # Align input with training feature order, add missing columns with 0
            input_df = input_df.reindex(columns=feature_names, fill_value=0)

            # Scale
            input_scaled = scaler.transform(input_df)

            # Predict
            prediction = model.predict(input_scaled)[0]

            result_text = "🟢 You are not likely experiencing depression." if prediction == 0 else "🔴 You may be at risk of depression."

            return render_template('result.html', prediction=result_text)

    except Exception as e:
        return f"⚠️ An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
