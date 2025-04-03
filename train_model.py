import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Load dataset
df = pd.read_csv("student-scores.csv")
print("Columns in dataset:", df.columns)  # Debugging: Check column names

# ✅ Calculate final score if missing
if 'final_score' not in df.columns:
    df['final_score'] = df[['math_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score']].mean(axis=1)

# ✅ Keep only required columns (Removed history_score)
selected_features = ['gender', 'part_time_job', 'absence_days', 'extracurricular_activities', 
                     'weekly_self_study_hours', 'career_aspiration', 
                     'math_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score']

df = df[selected_features + ['final_score']]  # ✅ Now final_score is present!

# ✅ Encode categorical variables
label_encoders = {}
for col in ['gender', 'part_time_job', 'extracurricular_activities', 'career_aspiration']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ✅ Split dataset
X = df[selected_features]
y = df['final_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train XGBoost Model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# ✅ Evaluate Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")

# ✅ Save Model & Label Encoders
pickle.dump(model, open("model/xgboost_model.pkl", "wb"))
pickle.dump(label_encoders, open("model/label_encoders.pkl", "wb"))

print("🎯 XGBoost Model Saved Successfully!")
