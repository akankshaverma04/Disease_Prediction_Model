# 🩺 Disease Prediction from Medical Data

## 📌 Overview

This project is a Machine Learning-based Disease Prediction System that predicts whether a patient is likely to have diabetes based on medical parameters such as glucose level, BMI, blood pressure, age, insulin, etc.

The project uses classification algorithms to analyze patient medical data and generate accurate predictions.

---

# 🎯 Objective

The main objective of this project is to:

- Predict the possibility of diabetes using patient medical data
- Apply machine learning classification techniques
- Compare different algorithms based on accuracy
- Analyze model performance using evaluation metrics

---

# 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# 📂 Dataset

### Dataset Used:
**Pima Indians Diabetes Dataset**

### Features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

### Target Variable:
- `0` → No Diabetes
- `1` → Diabetes Detected

---

# ⚙️ Machine Learning Algorithms Used

The following classification algorithms were implemented:

1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Machine (SVM)

---

# 🔄 Project Workflow

## 1️⃣ Import Required Libraries

Libraries for data analysis, visualization, preprocessing, and machine learning are imported.

---

## 2️⃣ Load Dataset

The dataset is loaded using Pandas.

```python
df = pd.read_csv("diabetes.csv")
```

---

## 3️⃣ Data Preprocessing

- Checked missing values
- Verified dataset structure
- Prepared clean data for training

```python
print(df.isnull().sum())
```

---

## 4️⃣ Feature Selection

Input features and target variable are separated.

```python
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
```

---

## 5️⃣ Train-Test Split

Dataset is divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

---

## 6️⃣ Feature Scaling

StandardScaler is used to normalize the feature values.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 7️⃣ Model Training

### Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train, y_train)
```

---

## 8️⃣ Prediction

```python
y_pred = rf.predict(X_test)
```

---

## 9️⃣ Model Evaluation

### Accuracy Score

```python
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
```

### Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

---

# 📊 Model Performance

| Model | Accuracy |
|------|------|
| Logistic Regression | Good |
| SVM | Better |
| Random Forest | Best (~88%) |

---

# ✅ Result

The Random Forest Classifier achieved the highest accuracy of approximately **88%** on the testing dataset.

The model successfully predicts whether a patient is diabetic or non-diabetic based on medical parameters.

---

# 📸 Sample Prediction

## Input

```python
sample = [[2,120,70,20,80,30.5,0.5,35]]
```

## Output

```python
0 → No Diabetes
1 → Diabetes Detected
```

---

# 📁 Project Structure

```bash
DiseasePrediction/
│
├── diseasePrediction.ipynb
├── diabetes.csv
├── README.md
└── requirements.txt
```

---

# ▶️ How to Run the Project

## Step 1: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Step 2: Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```bash
diseasePrediction.ipynb
```

---

# 🚀 Future Improvements

- Deploy using Flask or Streamlit
- Add graphical user interface (GUI)
- Add support for multiple diseases
- Improve accuracy using advanced ML models

---

# 👨‍💻 Author

**Akanksha Verma**

---

# ⭐ Conclusion

This project demonstrates how Machine Learning can be applied in healthcare to predict diseases using patient medical data. The system helps in early prediction and supports better medical decision-making.
