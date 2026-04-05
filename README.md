<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2E7D32,50:558B2F,100:1a0a2e&height=220&section=header&text=Crop%20Yield%20Prediction%20AI&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Machine%20Learning%20Driven%20Agricultural%20Intelligence&descAlignY=58&descSize=18&descColor=dddddd"/>

<img src="https://img.shields.io/badge/Best%20Model-KNN%20Regressor-2E7D32?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/R%C2%B2%20Score-98.5%25-7ee787?style=for-the-badge&logo=google-analytics&logoColor=white"/>
<img src="https://img.shields.io/badge/Dataset-28K%2B%20Records-58a6ff?style=for-the-badge&logo=kaggle&logoColor=white"/>
<img src="https://img.shields.io/badge/Framework-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>

</div>

---

## 📌 Executive Summary

Modern agriculture relies on data-driven decision-making to optimize global food production. This repository features an **End-to-End Machine Learning Pipeline** designed to accurately predict crop yields based on complex environmental and geographic factors. By evaluating multiple regression architectures, this project identifies highly precise models capable of turning raw climate and pesticide data into actionable agricultural intelligence.

---

## 🏗️ Data Pipeline Blueprint

The system processes raw tabular data through a robust feature engineering pipeline before feeding it into the predictive engine:

```text
RAW AGRICULTURAL DATA [28K+ Records]
  │
  ▼
┌──────────────────────────────────────────────┐
│  DATA ENGINEERING & PREPROCESSING            │
│  • Anomaly Detection (Invalid Rainfall Drop) │
│  • Standard Scaling (Temp, Rain, Pesticides) │
│  • One-Hot Encoding (Area, Crop Item)        │
└──────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────┐
│  REGRESSION MODELING ENGINE                  │
│  • Non-Linear: KNN (k=5), Decision Tree      │
│  • Linear: Ordinary, Ridge, Lasso            │
│  • Margin: Support Vector Regressor (SVR)    │
└──────────────────────────────────────────────┘
  │
  ▼
OUTPUT [Predicted Yield in hg/ha]
```


---

## 📂 Dataset Acquisition & Preparation

The dataset is sourced from Kaggle and contains over **28,000 records** spanning 101 countries and 10 major crop types. Due to file size constraints, the raw `.csv` is not hosted in this repository.

📥 **Download Dataset:** [**Crop Yield Prediction Dataset**](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)

### 🧹 Data Cleaning & Engineering Pipeline
The following code demonstrates how the raw data is prepared for the Machine Learning models:

```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('yield_df.csv')

# 1. Structural Cleaning
df.drop('Unnamed: 0', axis=1, errors='ignore', inplace=True)
df.drop_duplicates(inplace=True)

# 2. Handling Data Type Inconsistencies
# Specifically removing non-numeric noise from the rainfall column
def is_numeric(val):
    try:
        float(val)
        return True
    except:
        return False

df = df[df['average_rain_fall_mm_per_year'].apply(is_numeric)]
df['average_rain_fall_mm_per_year'] = df['average_rain_fall_mm_per_year'].astype(np.float64)

# 3. Feature Selection
# Focusing on the most impactful environmental and agricultural features
features = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item']
X = df[features]
y = df['hg/ha_yield']

print(f"Dataset ready for training. Total samples: {len(df)}")
```
---

## 📊 Model Evaluation & Performance

The pipeline evaluates 6 distinct regression algorithms using an 80/20 train-test split and **5-Fold Cross-Validation** to ensure the model generalizes well to unseen climate data.

| Rank | Model | R² Score | Cross-Val R² | MAE | RMSE |
|:---:|:--- |:---:|:---:|:---:|:---:|
| 🥇 | **KNN Regressor** | **0.985** | **0.980** | 4,611 | 10,396 |
| 🥈 | **Decision Tree** | **0.981** | **0.974** | 3,843 | 11,766 |
| 🥉 | Lasso Regression | 0.747 | 0.749 | 29,893 | 42,629 |
| 4 | Linear Regression | 0.747 | 0.749 | 29,907 | 42,630 |
| 5 | Ridge Regression | 0.747 | 0.749 | 29,864 | 42,631 |
| 6 | SVR | -0.206 | -0.191 | 57,810 | 93,125 |

> **💡 Key Insight:** Non-linear algorithms (**KNN & Decision Trees**) vastly outperformed traditional linear models. This confirms that the relationship between environmental variables (Rainfall/Temperature) and crop yield is highly complex and non-linear, which our top models captured with **>98% accuracy**.

---


---

## 💻 Full Pipeline Source Code

This automated pipeline handles the entire workflow: from data cleaning and feature scaling to multi-model training and real-time yield prediction.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. DATA PREPARATION
df = pd.read_csv('yield_df.csv')
df.drop('Unnamed: 0', axis=1, errors='ignore', inplace=True)
df.drop_duplicates(inplace=True)

# Define Features and Target
X = df[['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item']]
y = df['hg/ha_yield']

# 2. FEATURE ENGINEERING PIPELINE
# Scaling numerical data and encoding categorical labels (Country/Crop)
preprocessor = ColumnTransformer(
    transformers=[
        ('StandardScale', StandardScaler(), [0, 1, 2, 3]),
        ('OHE', OneHotEncoder(drop='first', handle_unknown='ignore'), [4, 5]),
    ],
    remainder='passthrough'
)

# 3. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_dummy = preprocessor.fit_transform(X_train)
X_test_dummy = preprocessor.transform(X_test)

# 4. INITIALIZING TOP-PERFORMING MODELS
models = {
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'DecisionTree': DecisionTreeRegressor()
}

# Training Models
for name, md in models.items():
    md.fit(X_train_dummy, y_train)

# 5. PRODUCTION INFERENCE FUNCTION
def prediction(model_name, Year, rainfall, pesticides, temp, Area, Item):
    """
    Predicts crop yield based on user input using the pre-trained pipeline.
    """
    # Create input dataframe
    features_df = pd.DataFrame(
        [[Year, rainfall, pesticides, temp, Area, Item]],
        columns=['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item']
    )
    
    # Apply the same transformations used in training
    transformed_features = preprocessor.transform(features_df)
    
    # Generate Prediction
    model = models[model_name]
    result = model.predict(transformed_features)
    return result[0]

# --- Example Usage ---
# predicted_yield = prediction('KNN', 1990, 1485.0, 121.0, 16.37, 'Albania', 'Maize')
# print(f"Predicted Yield: {predicted_yield:,.2f} hg/ha")
```

---

## 🛠️ Tech Stack

The following industry-standard tools and libraries were used to build, train, and evaluate this agricultural predictive pipeline:

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Seaborn-4C4C4C?style=flat-square&logo=python&logoColor=white" alt="Seaborn"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logo=python&logoColor=white" alt="Matplotlib"/>
</div>

| Tool | Role |
| :--- | :--- |
| **Scikit-Learn** | Core machine learning library for regression, scaling, and encoding. |
| **Pandas** | High-performance data structures for agricultural dataset manipulation. |
| **NumPy** | Numerical computing and multi-dimensional array processing. |
| **Seaborn / Matplotlib** | Statistical data visualization and model performance plotting. |
| **Python** | Primary programming language for the entire end-to-end pipeline. |

---


---

## 🚀 How to Run

Follow these steps to set up the environment and execute the crop yield prediction pipeline on your local machine.

### 1. Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone [https://github.com/mostafashazly7/crop-yield-prediction-ml.git](https://github.com/mostafashazly7/crop-yield-prediction-ml.git)
cd crop-yield-prediction-ml
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate
# Activate it (Mac/Linux)
source venv/bin/activate

# Install required libraries
pip install -r requirements.txt
```


3. Dataset Acquisition
- Due to storage limits, the dataset is not hosted directly in this repo.

- Download yield_df.csv from: Kaggle Crop Yield Dataset

- Move the downloaded file into the root folder of this project.

4. Running the Pipeline
- You can execute the project using the Jupyter Notebook or the Python source code:

- Interactive: Launch jupyter notebook and open Crop Yield Prediction project .ipynb.

- Inference: Use the prediction() function within your scripts to generate new yield forecasts:

# Example of running a prediction in your script
from source_code import prediction

result = prediction('KNN', 2013, 1012.0, 121.0, 16.5, 'Albania', 'Maize')
print(f"Predicted Yield: {result}")

---

## 👤 Author

**Mostafa Shazly** *Aspiring AI Engineer*

Throughout this project, I focused on building a clean, reproducible, and high-performance machine learning pipeline. My goal is to bridge the gap between complex agricultural data and practical, data-driven solutions for global food security.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/mostafa-shazly-148945314)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mostafashazly7)

<div align="center">
  <br>
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:2E7D32,50:558B2F,100:1a0a2e&height=100&section=footer"/>
  <br>
  <i>⭐ If you found this repository helpful, please consider leaving a star! ⭐</i>
</div>

