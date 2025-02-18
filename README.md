# **Sentiment Analysis for Food Reviews using Yelp® Review Dataset**

## **Project Overview**
This project applies **Natural Language Processing (NLP)** techniques along with **Machine Learning (ML) and Deep Learning (DL)** models to analyze sentiment in **Yelp restaurant reviews**. The objective is to classify reviews into **Positive, Neutral, or Negative** sentiments.

The project involves:
- **Preprocessing raw text data** using NLP techniques.
- **Feature extraction** using Bag of Words (BoW) and TF-IDF.
- **Training ML models** like Naïve Bayes, SVM, Logistic Regression, and XGBoost.
- **Training DL models** like RNN and LSTM for advanced text understanding.
- **Comparing model performance** and selecting the best models for sentiment classification.

---

## **Dataset Information**
The dataset consists of **Yelp restaurant reviews** containing:
- **Review Text** (`text`)
- **Star Ratings** (`stars`)
- **Additional Metadata** (useful votes, funny votes, timestamps)

---

## **NLP Preprocessing Steps**
Effective text preprocessing is crucial for improving model accuracy. The following **NLP techniques** were applied:

### **1️⃣ Text Cleaning**
- **HTML Tag Removal:** Removes unwanted `<html>` tags using BeautifulSoup.
- **Special Characters & Punctuation Removal:** Keeps only alphanumeric text.
- **Lowercasing:** Converts all text to lowercase for uniformity.

### **2️⃣ Tokenization**
- Breaks the text into individual words (**tokens**) for further processing.

### **3️⃣ Stopword Removal**
- Common stopwords (e.g., *the, is, at, which*) are removed to retain only meaningful words.
- **Customized Stopwords List:** Important sentiment-related words like **"not", "never", "bad", "good"** are retained.

### **4️⃣ Lemmatization**
- Converts words to their base form (e.g., *running → run, better → good*).
- Ensures consistency in word representation.

### **5️⃣ Feature Engineering**
To convert text into a numerical format for model training:
- **Bag of Words (BoW):** Counts word occurrences.
- **TF-IDF (Term Frequency-Inverse Document Frequency):** Assigns importance to words based on frequency.

### **6️⃣ Word Embeddings (Deep Learning)**
- **Tokenized sequences are padded** to ensure consistent input size for LSTM and RNN models.
- **Embedding layers are used** in deep learning models to capture contextual meaning.

---

## **Models Used**
This project compares **Machine Learning** and **Deep Learning** models for sentiment classification.

### **Machine Learning Models**
- **Naïve Bayes (BoW & TF-IDF)** – Baseline probabilistic model.
- **Support Vector Machine (SVM, BoW & TF-IDF)** – Strong linear classifier.
- **Logistic Regression (BoW & TF-IDF)** – Efficient and interpretable model.
- **XGBoost (BoW & TF-IDF)** – Advanced boosting-based model.

### **Deep Learning Models**
- **Recurrent Neural Network (RNN)** – Captures sequential relationships in text.
- **Long Short-Term Memory (LSTM)** – Improves context retention in longer texts.

---

## **Model Performance Comparison**
The following table presents the evaluation metrics for each model:

| **Model** | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|-----------|-------------|--------------|-----------|------------|
| **Naïve Bayes (BoW)** | 81.48% | 85.33% | 81.48% | 83.05% |
| **Naïve Bayes (TF-IDF)** | 85.05% | 81.84% | 85.05% | 82.36% |
| **SVM (BoW)** | 89.46% | 87.51% | 89.46% | 87.58% |
| **SVM (TF-IDF)** | 89.84% | 88.12% | 89.84% | 87.98% |
| **XGBoost (BoW)** | 87.88% | 85.99% | 87.88% | 86.23% |
| **XGBoost (TF-IDF)** | 88.09% | 86.23% | 88.09% | 86.46% |
| **Logistic Regression (BoW)** | 89.66% | 88.10% | 89.66% | 88.53% |
| **Logistic Regression (TF-IDF)** | **89.88%** | **88.40%** | **89.88%** | **88.76%** |
| **RNN** | 66.23% | 57.62% | 66.23% | 57.63% |
| **LSTM** | **89.78%** | **89.08%** | **89.78%** | **89.39%** |

### **Best Performing Models**
1. **LSTM** – Best for deep learning-based sentiment classification.
2. **Logistic Regression (TF-IDF)** – Best traditional ML model.
3. **SVM (TF-IDF)** – Reliable alternative to Logistic Regression.

---

## **How to Use**

```python
# Step 1: install dependencies
pip install -r requirements.txt

#Step 2: Load Pretrained Models
import joblib
from tensorflow.keras.models import load_model
logreg_tfidf = joblib.load("logreg_tfidf_model.pkl") # Load Logistic Regression Model
svm_tfidf = joblib.load("svm_tfidf_model.pkl") #Load SVM Model
lstm_model = load_model("lstm_model.h5") # Load LSTM

# Step 3: Predict Sentiment of New Reviews
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

sample_review = ["The food was terrible and overpriced."]

sample_tfidf = tfidf_vectorizer.transform(sample_review) # Convert text into numerical features

# Predict using ML models
logreg_pred = logreg_tfidf.predict(sample_tfidf)
svm_pred = svm_tfidf.predict(sample_tfidf)

# Predict using LSTM
sample_seq = tokenizer.texts_to_sequences(sample_review)
sample_pad = pad_sequences(sample_seq, maxlen=100)
lstm_pred = np.argmax(lstm_model.predict(sample_pad), axis=1)

print("Logistic Regression Prediction:", logreg_pred[0])
print("SVM Prediction:", svm_pred[0])
print("LSTM Prediction:", lstm_pred[0])
