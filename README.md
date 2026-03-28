# 🌾 Crop Yield Prediction باستخدام Machine Learning

## 📌 Overview
This project aims to predict crop yield using Machine Learning techniques based on environmental and agricultural factors such as rainfall, temperature, pesticide usage, country, and crop type.

---

## 📂 Dataset
The dataset is sourced from Kaggle:

- 📎 Crop Yield Prediction Dataset  
- Contains data for **100+ countries**  
- Total records: **28,000+**

### Features:
| Feature | Description |
|--------|------------|
| Area | Country |
| Item | Crop Type |
| Year | Year |
| average_rain_fall_mm_per_year | Annual Rainfall (mm) |
| pesticides_tonnes | Pesticide Usage (tonnes) |
| avg_temp | Average Temperature |
| hg/ha_yield | Yield (Target Variable) |

---

## 🧹 Data Preprocessing
- Removed unnecessary column (`Unnamed: 0`)
- Dropped duplicate rows
- Cleaned invalid rainfall values
- Converted data types where needed

---

## 📊 Exploratory Data Analysis (EDA)
- Distribution of crops across countries
- Yield comparison per country
- Yield comparison per crop
- Visualization using Seaborn & Matplotlib

---

## ⚙️ Machine Learning Pipeline
### 🔹 Data Splitting
- Train/Test split: 80% / 20%

### 🔹 Feature Engineering
- OneHotEncoding for categorical features (`Area`, `Item`)
- Standard Scaling for numerical features

### 🔹 Models Used
- Linear Regression
- Lasso Regression
- Ridge Regression
- Decision Tree Regressor 🌳
- K-Nearest Neighbors (KNN) 🤝
- Support Vector Regressor (SVR)

---

## 📈 Model Performance

| Model | R² Score | MAE | RMSE | CV R² |
|------|--------|------|------|------|
| KNN | 0.985 | 4611 | 10396 | 0.980 |
| Decision Tree | 0.981 | 3843 | 11766 | 0.974 |
| Lasso | 0.747 | 29893 | 42629 | 0.749 |
| Linear | 0.747 | 29907 | 42630 | 0.749 |
| Ridge | 0.747 | 29864 | 42631 | 0.749 |
| SVR | -0.206 | 57810 | 93125 | -0.191 |

---

## 🏆 Best Models
- ✅ KNN Regressor
- ✅ Decision Tree Regressor

These models achieved the highest accuracy and were selected for predictions.

---

## 🔮 Prediction Function
You can predict crop yield using:

```python
prediction(model_name, Year, rainfall, pesticides, temperature, Area, Item)
```

### Example:
```python
prediction('DecisionTree', 1990, 1485.0, 121.0, 16.37, 'Albania', 'Maize')
```

---

## 📊 Visualizations
- Average yield per crop
- Model comparison (R² vs CV R²)
- Country-wise yield distribution

---

## 🛠️ Technologies Used
- Python 🐍
- NumPy
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/mostafashazly7/crop-yield-prediction-ml
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebook or script

---

## 📌 Future Improvements
- Hyperparameter tuning
- Deploy as a web app (Flask / Streamlit)
- Add more real-world features (soil type, humidity, etc.)

---

## 👤 Author
**Mostafa Shazly**

- GitHub: https://github.com/mostafashazly7

---

## ⭐ Support
If you like this project, don't forget to ⭐ the repo!
