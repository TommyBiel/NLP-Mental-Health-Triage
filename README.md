# Mental Health NLP Classification: Automated Triage Engine

## 📌 Project Overview
Human moderators for online platforms cannot manually read and triage tens of thousands of user text logs in real-time. This project builds a Natural Language Processing (NLP) Machine Learning pipeline designed to automatically classify text into four categories: **Anxiety, Depression, Normal, and Suicidal**. 

The goal is to create a reliable baseline model that can flag high-risk texts (Suicidal/Depression) for immediate human intervention, optimizing the triage process and improving response times for users in crisis.

## 🛠️ Tech Stack & Tools
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Logistic Regression, TF-IDF Vectorization)
* **Data Visualization:** Matplotlib (Custom Grouped Bar Charts)

## ⚙️ The Data Pipeline & Methodology
1. **Text Preprocessing:** Cleaned and standardized a dataset of ~50,000 text logs (handling nulls, balancing classes, and text normalization).
2. **Feature Engineering (TF-IDF):** Converted raw text into mathematical matrices using Term Frequency-Inverse Document Frequency.
   * *Experimentation:* Tested `max_features` constraints (5k vs. 10k) and N-gram ranges (Unigrams vs. Bigrams) to capture deeper linguistic context.
3. **Model Training:** Built a Logistic Regression classification engine.
4. **Evaluation:** Prioritized **Recall** over overall accuracy to ensure high-risk texts were not missed by the radar. 

## 📊 Key Findings & Results
The optimized model achieved a **79% overall accuracy**. 

Through hyperparameter tuning, the model demonstrated the Law of Diminishing Returns: upgrading from single-word processing to Bigrams (two-word phrases) showed that a linear model (Logistic Regression) hits a mechanical ceiling at ~80% accuracy due to the complex, overlapping vocabulary between severe depression and suicidal ideation. 

### Performance Metrics (Recall by Status):
* **Normal:** 93.0%
* **Anxiety:** 86.5%
* **Suicidal:** 70.0%
* **Depression:** 64.6%

<img width="3012" height="1176" alt="bigram_linear_regression" src="https://github.com/user-attachments/assets/c6f1da8f-dd09-4917-9952-1eb259e612bd" />

## 🚀 Next Steps & Future Improvements
To break past the 80% accuracy ceiling established by this baseline model, future iterations would require abandoning linear models in favor of:
* **Ensemble Methods:** Random Forest or XGBoost.
* **Deep Learning:** Implementing pre-trained transformer models (like BERT) to better understand the sequential context and emotional sentiment of the text.
