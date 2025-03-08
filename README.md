# Spam Email Prediction using Machine Learning

This project is a **Spam Email Prediction System** that uses **Machine Learning** to classify emails as either "Spam" or "Ham" (not spam). It utilizes **Logistic Regression** as the classification model and **TF-IDF (Term Frequency-Inverse Document Frequency)** for feature extraction.

---

## 📌 Features
- Loads email data from a CSV file.
- Preprocesses the dataset (handles missing values and encodes labels).
- Splits the data into training and testing sets.
- Converts text data into numerical features using **TF-IDF Vectorization**.
- Trains a **Logistic Regression** model.
- Evaluates model accuracy on training and test datasets.
- Provides a prediction system for user input email messages.

---

## 🛠️ Technologies Used
- Python 🐍
- Pandas (for data handling)
- NumPy (for numerical operations)
- Scikit-learn (for Machine Learning)
- Logistic Regression (Classification Model)
- TF-IDF Vectorizer (Feature Extraction)

---

## 📂 Dataset
The dataset used is `mail_data.csv`, which consists of two columns:
- **Category**: Labels (Spam = 0, Ham = 1)
- **Message**: Email content

---

## 📊 Model Performance
The model achieves an accuracy of approximately:
- **Training Data Accuracy**: `~X%`
- **Test Data Accuracy**: `~Y%`

(Replace X and Y with actual accuracy values after training the model.)

---

## 🔍 How It Works
1. The dataset is loaded and preprocessed.
2. Text messages are converted into numerical vectors using **TF-IDF**.
3. The **Logistic Regression** model is trained on this transformed data.
4. The model predicts whether a given email is **Spam or Ham**.

---

## 📝 Example Usage
```python
input_mail = ["Congratulations! You have won a lottery. Claim now!"]
input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)
print("Spam Mail" if prediction[0] == 0 else "Ham Mail")
```

---



## 🤝 Contributing
Feel free to **fork** this repository and make improvements. If you have any suggestions, open an issue or submit a pull request!



