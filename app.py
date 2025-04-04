from flask import Flask, render_template, request
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import json

app = Flask(__name__)

# ✅ Load model and encoders
with open("model/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# ✅ Create graph folder if not exist
os.makedirs("static/graphs", exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Fetch form data
        form = request.form
        input_data = {
            "gender": form.get("gender", "Unknown"),
            "part_time_job": form.get("part_time_job", "No"),
            "absence_days": float(form.get("absence_days", 0)),
            "extracurricular_activities": form.get("extracurricular", "No"),
            "weekly_self_study_hours": float(form.get("self_study", 0)),
            "career_aspiration": form.get("career_aspiration", "None"),
            "math_score": float(form.get("math_score", 0)),
            "physics_score": float(form.get("physics_score", 0)),
            "chemistry_score": float(form.get("chemistry_score", 0)),
            "biology_score": float(form.get("biology_score", 0)),
            "english_score": float(form.get("english_score", 0)),
        }

        # ✅ Encode categorical features
        def encode(val, col):
            return label_encoders[col].transform([val])[0] if val in label_encoders[col].classes_ else 0

        encoded = {
            "gender": encode(input_data["gender"], "gender"),
            "part_time_job": encode(input_data["part_time_job"], "part_time_job"),
            "extracurricular_activities": encode(input_data["extracurricular_activities"], "extracurricular_activities"),
            "career_aspiration": encode(input_data["career_aspiration"], "career_aspiration"),
        }

        X_input = np.array([[
            encoded["gender"],
            encoded["part_time_job"],
            input_data["absence_days"],
            encoded["extracurricular_activities"],
            input_data["weekly_self_study_hours"],
            encoded["career_aspiration"],
            input_data["math_score"],
            input_data["physics_score"],
            input_data["chemistry_score"],
            input_data["biology_score"],
            input_data["english_score"]
        ]])

        prediction = model.predict(X_input)[0]

        # ✅ Graph 1: Radar Chart
        subjects = ['Math', 'Physics', 'Chemistry', 'Biology', 'English']
        scores = [input_data[f"{s.lower()}_score"] for s in subjects]

        fig, ax = plt.subplots(figsize=(5,5), subplot_kw={'polar': True})
        angles = np.linspace(0, 2 * np.pi, len(subjects), endpoint=False).tolist()
        scores += scores[:1]
        angles += angles[:1]

        ax.plot(angles, scores, 'o-', linewidth=2)
        ax.fill(angles, scores, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(subjects)
        ax.set_ylim(0, 100)
        plt.title("Subject-wise Performance Radar")
        radar_path = "static/graphs/radar.png"
        plt.savefig(radar_path)
        plt.close()

        # ✅ Graph 2: Feature Importance
        importance = model.feature_importances_
        features = ['Gender', 'Job', 'Absence', 'Extra', 'StudyHrs', 'Career', 'Math', 'Physics', 'Chemistry', 'Biology', 'English']
        fig, ax = plt.subplots()
        ax.barh(features, importance)
        plt.title("Feature Importance")
        plt.tight_layout()
        feature_path = "static/graphs/importance.png"
        plt.savefig(feature_path)
        plt.close()

        # ✅ Graph 3: Performance Trend (simulate based on scores)
        trends = [input_data[f"{s.lower()}_score"] for s in ['math', 'physics', 'chemistry', 'biology', 'english']]
        fig, ax = plt.subplots()
        ax.plot(subjects, trends, marker='o')
        plt.title("Performance Trend")
        plt.ylim(0, 100)
        trend_path = "static/graphs/trend.png"
        plt.savefig(trend_path)
        plt.close()

        # ✅ Career-Specific Insights
        career = input_data["career_aspiration"]
        important_subjects = {
            "Engineer": ["Math", "Physics"],
            "Doctor": ["Biology", "Chemistry"],
            "Business": ["Math", "English"],
            "Other": ["English"]
        }
        suggested = important_subjects.get(career, ["English"])

        # ✅ Study Tips
        weak_subjects = [sub for sub, score in zip(subjects, scores[:-1]) if score < 50]
        tips = {
            "Math": "Practice daily problems and revise formulas.",
            "Physics": "Focus on concepts and solve previous year questions.",
            "Chemistry": "Revise reactions and practice numericals.",
            "Biology": "Use flashcards and diagrams to retain terms.",
            "English": "Read books and practice writing essays."
        }
        suggestions = [tips[sub] for sub in weak_subjects]

        return render_template("result.html",
                               prediction=round(prediction, 2),
                               radar=radar_path,
                               feature=feature_path,
                               trend=trend_path,
                               suggested_subjects=suggested,
                               study_tips=suggestions)

    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
