<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=soft&color=0:ffffff,100:f0f4f8&height=180&section=header&text=Vanishree&fontSize=52&fontColor=1a1a2e&fontAlignY=45&desc=Data%20Science%20Engineer%20%7C%20ECE%20%7C%20VTU%202022&descAlignY=65&descSize=16&descColor=4a4a6a" />

<h3>— Building clean, accurate, end-to-end ML systems —</h3>

<p>
<b>Data Science Engineer (ECE) | VTU</b> — Builds end-to-end ML pipelines across classification,
clustering, and NLP — delivering systems that hit <b>87–88% accuracy</b> on real-world problems
like loan risk, fake news, and student outcomes.
</p>

[

![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)

](https://linkedin.com/in/-vanishree-)
[

![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)

](https://github.com/Vanishree)
[

![Email](https://img.shields.io/badge/Email-EA4335?style=flat-square&logo=gmail&logoColor=white)

](mailto:vanishreens2005@gmail.com)

</div>

---

## Tech Stack



![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)




![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)




![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)




![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)




![NLTK](https://img.shields.io/badge/NLTK-2e7d32?style=flat-square&logo=python&logoColor=white)




![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logo=python&logoColor=white)




![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)




![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=black)





![Supervised Learning](https://img.shields.io/badge/Supervised_Learning-6a1b9a?style=flat-square&logoColor=white)




![Unsupervised Learning](https://img.shields.io/badge/Unsupervised_Learning-283593?style=flat-square&logoColor=white)




![NLP](https://img.shields.io/badge/NLP-c62828?style=flat-square&logoColor=white)




![Classification](https://img.shields.io/badge/Classification-00695c?style=flat-square&logoColor=white)




![Clustering](https://img.shields.io/badge/Clustering-e65100?style=flat-square&logoColor=white)



---

## Projects

---

### 01 — Student Performance Prediction System

> Python · Pandas · Scikit-learn · Matplotlib · Jupyter Notebook

Predicts whether a student will Pass or Fail using supervised ML trained on
attendance, study hours, previous marks, and internal scores.

**Pipeline:**
`Data Collection` → `Preprocessing` → `Feature Selection` → `Model Training` → `Evaluation`

**Results:**

| Model | Accuracy |
|---|---|
| Random Forest | **88%** ✅ |
| SVM | 85% |
| Logistic Regression | 82% |
| Decision Tree | 78% |
| Naive Bayes | 75% |

<details>
<summary>📄 View Full Project Code</summary>

```python
# ============================================================
# PROJECT 1 — Student Performance Prediction System
# VTU 2022 Scheme | Data Science Lab | Dept. of ECE
# ============================================================

# STEP 1 — Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# STEP 2 — Create Sample Dataset
np.random.seed(42)
n = 200

data = pd.DataFrame({
    'Study_Hours':     np.random.uniform(1, 10, n),
    'Attendance':      np.random.uniform(50, 100, n),
    'Previous_Marks':  np.random.uniform(40, 100, n),
    'Assignments':     np.random.uniform(50, 100, n),
    'Internal_Marks':  np.random.uniform(40, 100, n),
})

data['Final_Result'] = (
    (data['Study_Hours'] * 3 +
     data['Attendance'] * 0.3 +
     data['Previous_Marks'] * 0.4 +
     data['Internal_Marks'] * 0.3 +
     np.random.normal(0, 5, n)) > 50
).astype(int)

print("Dataset Shape:", data.shape)
print(data.head())

# STEP 3 — EDA
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.scatterplot(data=data, x='Study_Hours', y='Previous_Marks', hue='Final_Result',
                palette={0: 'red', 1: 'green'})
plt.title('Study Hours vs Previous Marks')
plt.legend(labels=['Fail', 'Pass'])

plt.subplot(1, 2, 2)
sns.histplot(data=data, x='Attendance', hue='Final_Result',
             palette={0: 'red', 1: 'green'}, bins=20)
plt.title('Attendance Distribution')
plt.tight_layout()
plt.show()

# STEP 4 — Preprocessing
X = data[['Study_Hours', 'Attendance', 'Previous_Marks', 'Internal_Marks']]
y = data['Final_Result']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# STEP 5 — Train and Compare Models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Naive Bayes':         GaussianNB(),
    'SVM':                 SVC(random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name:25s} → Accuracy: {acc*100:.2f}%")

# STEP 6 — Accuracy Comparison Chart
plt.figure(figsize=(10, 5))
bars = plt.bar(results.keys(), [v * 100 for v in results.values()],
               color=['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f'])
plt.ylim(60, 100)
plt.title('Model Accuracy Comparison — Student Performance Prediction', fontsize=13)
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=15)
for bar, val in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val*100:.1f}%', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# STEP 7 — Best Model Evaluation
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

print("\n--- Random Forest Classification Report ---")
print(classification_report(y_test, y_pred_best, target_names=['Fail', 'Pass']))

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
plt.title('Confusion Matrix — Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

print("\nFinal Accuracy (Random Forest):", accuracy_score(y_test, y_pred_best) * 100, "%")
