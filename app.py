import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import streamlit as st
import pickle

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv('D:\ExcelR\Assignment\Logistic Regression\Titanic_train.csv')

data = load_data()

st.title("Logistic Regression: Titanic Survival Prediction")

# Exploratory Data Analysis
st.subheader("Dataset Overview")
st.dataframe(data.head())
st.write("Summary Statistics:", data.describe())
st.write("Missing Values:", data.isnull().sum())

# Handle Missing Values
imputer = SimpleImputer(strategy='median')
data['Age'] = imputer.fit_transform(data[['Age']])
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Encode Categorical Variables
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

#Histogram
st.subheader("visualizations such as histograms, box plots, or pair plots to visualize the distributions and relationships between features.")
st.write("1.Histogram Chart")
data['Age'].hist(bins=30,color='orange',edgecolor ='black')
plt.title('Histograms of Age')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
st.pyplot(plt)

#Box plot
st.write('2.Box plot')
sns.boxplot(x = data['Sex'],y= data['Age'])
plt.title('Boxplot of Age and Sex')
st.pyplot(plt)

#Pair plot
st.write("3.Pair plot")
sns.pairplot(data[['Fare','Parch']])
plt.show()
st.pyplot(plt)

# Feature and Target Selection
X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex']]
y = data['Survived']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Model Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

st.subheader("Model Evaluation")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.text(classification_report(y_test, y_pred))
st.write("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
st.pyplot(plt)

# Streamlit Form for Prediction
st.subheader("Make a Prediction")
pclass = st.number_input("Passenger Class (1, 2, 3)", min_value=1, max_value=3, value=1)
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
sex = st.radio("Sex", ["Male", "Female"])

sex = 1 if sex == "Male" else 0

if st.button("Predict Survival"):
    input_data = np.array([[pclass, age, sibsp, parch, fare, sex]])
    prediction = model.predict(input_data)
    st.write("Prediction: Survived" if prediction[0] == 1 else "Prediction: Did Not Survive")