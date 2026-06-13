# 🚨 Comment Toxicity Classification using Ensemble ML Models

A multi-class Natural Language Processing (NLP) system that classifies online comments into four toxicity levels: **Safe, Unsafe, Toxic, and Extremely Toxic**. The project combines advanced feature engineering with ensemble machine learning techniques to improve moderation accuracy on large-scale user-generated content.

---

## 📌 Project Overview

* **Dataset Size:** 198,000 training samples and 102,000 test samples
* **Task:** Multi-class text classification (4 classes)
* **Best Model:** Weighted Soft-Voting Ensemble (LightGBM + XGBoost)
* **Final Accuracy:** 91.58%
* **Macro F1-Score:** 0.8204

---

## 🎯 Problem Statement

Online platforms receive millions of comments daily, making manual moderation difficult. This project aims to automatically identify and categorize comments based on their toxicity level, helping moderation systems prioritize harmful content efficiently.

### Toxicity Categories

| Label | Category        |
| ----- | --------------- |
| 0     | Safe            |
| 1     | Unsafe          |
| 2     | Toxic           |
| 3     | Extremely Toxic |

---

## 🔧 Feature Engineering

### 1. Word-Level TF-IDF

* 15,000 features
* Unigrams and Bigrams (1–2 grams)
* Captures contextual word patterns and semantic information

### 2. Character-Level TF-IDF

* 3,000 features
* Character n-grams (3–5 grams)
* Detects misspellings, obfuscated profanity, and abusive word variations

### 3. Behavioral & Linguistic Features

Additional handcrafted features were extracted to improve classification performance:

* Upvote count
* Downvote count
* Vote ratio
* Comment length
* Lexical diversity
* Profanity count
* Emoticon count
* Capitalization ratio
* Punctuation frequency
* Special character usage

---

## 🤖 Models Evaluated

Models were trained using a stratified 80-20 train-validation split.

| Model               | Macro F1-Score |
| ------------------- | -------------- |
| Logistic Regression | 0.6279         |
| XGBoost             | 0.8054         |
| LightGBM            | 0.8201         |

**Best Standalone Model:** LightGBM

---

## ⚙️ Model Optimization

### Threshold Tuning

Class-specific probability thresholds were optimized to improve recall for minority classes and maximize Macro F1-Score.

### Ensemble Learning

A weighted soft-voting ensemble was built using:

* LightGBM
* XGBoost

The ensemble combines probability outputs from both models, resulting in improved robustness and balanced class performance.

---

## 📈 Final Results

| Metric          | Score  |
| --------------- | ------ |
| Accuracy        | 91.58% |
| Macro Precision | 82.10% |
| Macro Recall    | 82.19% |
| Macro F1-Score  | 82.04% |

### Classification Report

```text
              precision    recall  f1-score   support

Safe            0.9804    0.9474    0.9636     22835
Unsafe          0.7534    0.8206    0.7856      3183
Toxic           0.8671    0.9089    0.8875     12488
Extremely Toxic 0.6830    0.6106    0.6448      1094

accuracy                              0.9158     39600
macro avg       0.8210    0.8219    0.8204     39600
weighted avg    0.9182    0.9158    0.9165     39600
```

---

## 🛠️ Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* XGBoost

### Techniques Used

* Natural Language Processing (NLP)
* TF-IDF Vectorization
* Feature Engineering
* Hyperparameter Tuning (RandomizedSearchCV)
* Class Weight Balancing
* Threshold Optimization
* Ensemble Learning

---

## 📂 Repository Structure

```text
├── comment_toxicity_classifier.ipynb
├── README.md
├── data/

```

### Main File

* `comment_toxicity_classifier.ipynb` – Complete workflow including data preprocessing, feature engineering, model training, evaluation, threshold tuning, and ensemble creation.

---

## 👨‍💻 Author

**Robin Gupta**

BS in Data Science and Applications
Indian Institute of Technology Madras

LinkedIn: https://linkedin.com/in/robin-gupta-3b141a257

GitHub: https://github.com/RobinGupta2003
