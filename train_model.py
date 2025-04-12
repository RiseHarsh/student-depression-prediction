import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv('cleaned_dataset.csv')

# 2. Handle missing/invalid data
df['Financial Stress'] = pd.to_numeric(df['Financial Stress'], errors='coerce')
df.dropna(inplace=True)

# 3. Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 4. Split features and target
X = df.drop('Depression', axis=1)
y = df['Depression']

# 5. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 8. Evaluate Model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Logistic Regression Accuracy: {acc:.4f}\n")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 9. Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 10. Save model, scaler, encoders
joblib.dump(model, 'model/logistic_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(label_encoders, 'model/encoders.pkl')
joblib.dump(X.columns.tolist(), 'model/feature_names.pkl')  # 🔥 Add this line

print("\n✅ Logistic Regression model, scaler, encoders, and feature names saved successfully.")

# print("\n✅ Logistic Regression model, scaler, and encoders saved successfully.")
