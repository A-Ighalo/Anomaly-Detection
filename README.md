#  Loyalty Program Fraud Detection System

A comprehensive anomaly detection system designed to identify fraudulent activities in loyalty and promotion programs using machine learning techniques.

##  Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Results](#results)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

##  Overview

This project implements both **supervised** and **unsupervised** machine learning approaches to detect anomalous patterns in credit card transactions that could indicate loyalty program abuse. The system is designed to help financial institutions prevent fraud while maintaining customer trust.

### Key Objectives
- Detect unusual patterns of reward accumulation or redemption
- Prevent loyalty program abuse
- Reduce financial losses from fraudulent transactions
- Automate fraud detection processes

##  Dataset

- **Source**: Credit Card Fraud Detection Dataset (Kaggle)
- **Size**: 284,807 transactions over 2 days
- **Features**: 31 variables including anonymized features (V1-V28), Amount, Time, and Class
- **Target**: Binary classification (0 = Normal, 1 = Fraud)
- **Class Distribution**: Highly imbalanced (0.17% fraudulent transactions)

##  Features

### Data Engineering
- **Reward Points Simulation**: 5% cashback calculation
- **User ID Assignment**: Random user mapping for behavioral analysis
- **Time-based Features**: Hour extraction, time period categorization
- **Amount Categorization**: Transaction amount binning
- **Feature Scaling**: StandardScaler for numerical features

### Machine Learning Models

#### Supervised Learning
- **Random Forest Classifier** with class balancing
- **LightGBM** for gradient boosting
- **XGBoost** for extreme gradient boosting

#### Unsupervised Anomaly Detection
- **Isolation Forest** for outlier detection
- **Local Outlier Factor (LOF)** for density-based anomalies
- **One-Class SVM** for boundary-based detection
- **DBSCAN** for clustering-based anomalies

##  Installation

### Prerequisites
```bash
Python 3.7+
pip or conda package manager
```

### Clone Repository
```bash
git clone https://github.com/yourusername/loyalty-fraud-detection.git
cd loyalty-fraud-detection
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.2.0
xgboost>=1.4.0
matplotlib>=3.4.0
seaborn>=0.11.0
streamlit>=1.0.0
joblib>=1.0.0
kagglehub>=0.1.0
```

##  Usage

### 1. Data Preparation
```python
import pandas as pd
import kagglehub

# Download dataset
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
df = pd.read_csv(path + '/creditcard.csv')
```

### 2. Feature Engineering
```python
# Create reward points
df['reward_points'] = df['Amount'] * 0.05

# Generate user IDs
np.random.seed(0)
df['user_id'] = np.random.randint(0, 1000, df.shape[0])

# Extract time features
df['hour_24'] = (df['Time'] % 86400) // 3600
```

### 3. Model Training
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
```

### 4. Web Application - in progress
```bash
streamlit run app.py
```

##  Model Performance

| Model | ROC AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| **XGBoost** | **0.986** | **0.97** | **0.80** | **0.88** |
| LightGBM | 0.973 | 0.96 | 0.82 | 0.88 |
| Random Forest | 0.958 | 0.93 | 0.84 | 0.88 |
| Isolation Forest | 0.952 | 0.12 | 0.59 | 0.21 |
| One-Class SVM | 0.948 | 0.08 | 0.53 | 0.14 |
| LOF | 0.672 | 0.02 | 0.09 | 0.03 |
| DBSCAN | 0.621 | 0.00 | 1.00 | 0.00 |

##  Project Structure

```
loyalty-fraud-detection/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── anomaly_detection_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluation.py
│
├── models/
│   └── rf_model.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
```

##  Results

### Key Findings
- **Peak Transaction Time**: 9 PM shows highest transaction volume
- **Spending Patterns**: Majority of users spend less than €500
- **Fraud Timing**: Fraudulent transactions more common in afternoon hours
- **Best Model**: XGBoost achieved highest ROC AUC of 0.986

### Visualizations
- Transaction distribution by hour
- Amount spending patterns
- ROC curve comparisons
- Fraud detection by time periods

##  Deployment

### Streamlit Web App
The project includes a user-friendly web interface that allows:
- Upload transaction datasets (CSV format)
- Real-time fraud prediction
- Interactive results visualization
- Model performance metrics

### Running the App
```bash
streamlit run app.py
```

### Features
-  File upload for transaction data
-  Fraud detection results
-  Prediction probabilities
-  Downloadable results

##  Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black src/
flake8 src/
```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔮 Future Enhancements

- [ ] Real-time streaming data processing
- [ ] Deep learning models (Autoencoders, LSTM)
- [ ] Advanced feature engineering with behavioral patterns
- [ ] Integration with banking APIs
- [ ] Mobile application development
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)
