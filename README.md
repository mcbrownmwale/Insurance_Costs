# Medical Insurance Cost Prediction

## 📌 Overview
This project analyzes factors affecting medical insurance costs and builds a predictive model using linear regression. The model predicts insurance charges based on patient demographics, with a focus on age and smoking status.

## 🏥 Dataset
**Source:** [Kaggle Medical Costs Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)  
**Features:**
- `age`: Patient age
- `sex`: Gender (male/female)
- `bmi`: Body mass index
- `children`: Number of dependents
- `smoker`: Smoking status (yes/no)
- `region`: Geographic region
- `charges`: Individual medical costs (target variable)

## 🔧 Data Preparation
```python
# Key transformations
df['log_charges'] = np.log2(df['charges'])  # Log transformation
df = pd.get_dummies(df, columns=['smoker'])  # One-hot encoding
