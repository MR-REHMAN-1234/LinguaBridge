import pandas as pd
import numpy as np
import re
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("   LINGUABRIDGE - MODEL TRAINING")
print("=" * 60)

# ─────────────────────────────────────────
# HELPER FUNCTION
# ─────────────────────────────────────────
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

# ═════════════════════════════════════════
# MODEL 1: LANGUAGE DETECTION
# ═════════════════════════════════════════
print("\n🌍 MODEL 1: LANGUAGE DETECTION")
print("-" * 40)

# Load
df1 = pd.read_csv('data/Language_Detection_Clean.csv')
print(f"✅ Loaded: {df1.shape[0]} rows")

# Preprocessing
df1['Text_Clean'] = df1['Text'].apply(preprocess)

# Label Encoding
le1 = LabelEncoder()
y1 = le1.fit_transform(df1['Language'])
print(f"✅ Classes: {list(le1.classes_)}")

# TF-IDF
tfidf1 = TfidfVectorizer(max_features=5000)
X1 = tfidf1.fit_transform(df1['Text_Clean'])

# Train/Validation/Test Split (80/5/15)
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.15, random_state=42
)
X1_train, X1_val, y1_train, y1_val = train_test_split(
    X1_train, y1_train, test_size=0.06, random_state=42
)

print(f"\n📊 Split:")
print(f"   Training:   {X1_train.shape[0]} rows (80%)")
print(f"   Validation: {X1_val.shape[0]} rows (5%)")
print(f"   Testing:    {X1_test.shape[0]} rows (15%)")

# Train — Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X1_train, y1_train)

# Validation Accuracy
val_acc = accuracy_score(y1_val, nb_model.predict(X1_val))
print(f"\n📌 Validation Accuracy: {val_acc*100:.2f}%")

# Test Accuracy
test_acc = accuracy_score(y1_test, nb_model.predict(X1_test))
print(f"📌 Test Accuracy:       {test_acc*100:.2f}%")

# Compare with Random Forest
rf1 = RandomForestClassifier(n_estimators=100, random_state=42)
rf1.fit(X1_train, y1_train)
rf_acc = accuracy_score(y1_test, rf1.predict(X1_test))
print(f"\n📊 Model Comparison:")
print(f"   Naive Bayes:   {test_acc*100:.2f}% ← Our Choice")
print(f"   Random Forest: {rf_acc*100:.2f}%")

# Classification Report
print(f"\n📋 Classification Report (Naive Bayes):")
print(classification_report(
    y1_test,
    nb_model.predict(X1_test),
    target_names=le1.classes_
))

# Save
pickle.dump(nb_model, open('models/language_model.pkl', 'wb'))
pickle.dump(tfidf1,   open('models/tfidf_lang.pkl', 'wb'))
pickle.dump(le1,      open('models/lang_encoder.pkl', 'wb'))
print("✅ Language model saved!")

# Quick Test
test_texts = ["Hello how are you", "Bonjour comment allez vous", "Hola como estas"]
X_test_quick = tfidf1.transform([preprocess(t) for t in test_texts])
predictions = le1.inverse_transform(nb_model.predict(X_test_quick))
print(f"\n📌 Quick Test:")
for text, pred in zip(test_texts, predictions):
    print(f"   '{text}' → {pred}")

# ═════════════════════════════════════════
# MODEL 2: SENTIMENT ANALYSIS
# ═════════════════════════════════════════
print("\n\n😊 MODEL 2: SENTIMENT ANALYSIS")
print("-" * 40)

# Load
df2 = pd.read_csv('data/Sentiment_Clean.csv')
print(f"✅ Loaded: {df2.shape[0]} rows")

# Preprocessing
df2['text_clean'] = df2['text'].apply(preprocess)

# TF-IDF
tfidf2 = TfidfVectorizer(max_features=5000)
X2 = tfidf2.fit_transform(df2['text_clean'])
y2 = df2['sentiment']

# Train/Validation/Test Split (80/10/10)
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.10, random_state=42
)
X2_train, X2_val, y2_train, y2_val = train_test_split(
    X2_train, y2_train, test_size=0.11, random_state=42
)

print(f"\n📊 Split:")
print(f"   Training:   {X2_train.shape[0]} rows (80%)")
print(f"   Validation: {X2_val.shape[0]} rows (10%)")
print(f"   Testing:    {X2_test.shape[0]} rows (10%)")

# Train — Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X2_train, y2_train)

# Validation
val_acc2 = accuracy_score(y2_val, lr_model.predict(X2_val))
print(f"\n📌 Validation Accuracy: {val_acc2*100:.2f}%")

# Test
test_acc2 = accuracy_score(y2_test, lr_model.predict(X2_test))
print(f"📌 Test Accuracy:       {test_acc2*100:.2f}%")

# Compare with Naive Bayes
nb2 = MultinomialNB()
nb2.fit(X2_train, y2_train)
nb_acc2 = accuracy_score(y2_test, nb2.predict(X2_test))
print(f"\n📊 Model Comparison:")
print(f"   Logistic Regression: {test_acc2*100:.2f}% ← Our Choice")
print(f"   Naive Bayes:         {nb_acc2*100:.2f}%")

# Classification Report
print(f"\n📋 Classification Report:")
print(classification_report(
    y2_test, lr_model.predict(X2_test),
    target_names=['Negative', 'Positive']
))

# Save
pickle.dump(lr_model, open('models/sentiment_model.pkl', 'wb'))
pickle.dump(tfidf2,   open('models/tfidf_sentiment.pkl', 'wb'))
print("✅ Sentiment model saved!")

# Quick Test
test_tweets = [
    "I love this so much!",
    "This is terrible I hate it",
    "Amazing day today!"
]
X_test_q2 = tfidf2.transform([preprocess(t) for t in test_tweets])
preds2 = lr_model.predict(X_test_q2)
labels2 = {0: 'Negative', 1: 'Positive'}
print(f"\n📌 Quick Test:")
for text, pred in zip(test_tweets, preds2):
    print(f"   '{text}' → {labels2[pred]}")

# ═════════════════════════════════════════
# MODEL 3: TONE CLASSIFIER
# ═════════════════════════════════════════
print("\n\n🎭 MODEL 3: TONE CLASSIFIER")
print("-" * 40)

# Load
df3 = pd.read_csv('data/tone_dataset.csv')
print(f"✅ Loaded: {df3.shape[0]} rows")

# Preprocessing
df3['sentence_clean'] = df3['sentence'].apply(preprocess)

# Label Encoding
le3 = LabelEncoder()
y3 = le3.fit_transform(df3['label'])

# TF-IDF
tfidf3 = TfidfVectorizer(max_features=500)
X3 = tfidf3.fit_transform(df3['sentence_clean'])

# Cross Validation (Dataset chhota hai — 203 rows)
print(f"\n📊 5-Fold Cross Validation:")
lr3 = LogisticRegression(max_iter=1000, random_state=42)
cv_scores = cross_val_score(lr3, X3, y3, cv=5)
print(f"   Fold 1: {cv_scores[0]*100:.2f}%")
print(f"   Fold 2: {cv_scores[1]*100:.2f}%")
print(f"   Fold 3: {cv_scores[2]*100:.2f}%")
print(f"   Fold 4: {cv_scores[3]*100:.2f}%")
print(f"   Fold 5: {cv_scores[4]*100:.2f}%")
print(f"   Average: {cv_scores.mean()*100:.2f}%")

# Final Train on full data
lr3.fit(X3, y3)
final_acc = accuracy_score(y3, lr3.predict(X3))
print(f"\n📌 Final Training Accuracy: {final_acc*100:.2f}%")

# Compare with SVM
from sklearn.svm import SVC
svm3 = SVC(random_state=42)
svm_scores = cross_val_score(svm3, X3, y3, cv=5)
print(f"\n📊 Model Comparison:")
print(f"   Logistic Regression: {cv_scores.mean()*100:.2f}% ← Our Choice")
print(f"   SVM:                 {svm_scores.mean()*100:.2f}%")

# Save
pickle.dump(lr3,   open('models/tone_model.pkl', 'wb'))
pickle.dump(tfidf3, open('models/tfidf_tone.pkl', 'wb'))
pickle.dump(le3,   open('models/tone_encoder.pkl', 'wb'))
print("✅ Tone model saved!")

# Quick Test
test_sentences = [
    "I would like to request your assistance.",
    "Hey wassup dude!",
    "Please be advised that the meeting is scheduled.",
    "omg that is so cool lol"
]
X_test_q3 = tfidf3.transform([preprocess(t) for t in test_sentences])
preds3 = le3.inverse_transform(lr3.predict(X_test_q3))
print(f"\n📌 Quick Test:")
for text, pred in zip(test_sentences, preds3):
    print(f"   '{text}'")
    print(f"   → {pred.upper()}")

# ═════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════
print("\n" + "=" * 60)
print("   📊 FINAL SUMMARY")
print("=" * 60)
print(f"""
Model 1 — Language Detection:
   Algorithm:  Naive Bayes
   Test Acc:   {test_acc*100:.2f}%
   Split:      80/5/15

Model 2 — Sentiment Analysis:
   Algorithm:  Logistic Regression
   Test Acc:   {test_acc2*100:.2f}%
   Split:      80/10/10

Model 3 — Tone Classifier:
   Algorithm:  Logistic Regression
   CV Acc:     {cv_scores.mean()*100:.2f}%
   Method:     5-Fold Cross Validation

Saved Models:
   ✅ models/language_model.pkl
   ✅ models/sentiment_model.pkl
   ✅ models/tone_model.pkl
   ✅ models/tfidf_lang.pkl
   ✅ models/tfidf_sentiment.pkl
   ✅ models/tfidf_tone.pkl
   ✅ models/lang_encoder.pkl
   ✅ models/tone_encoder.pkl
""")
print("=" * 60)
print("   ✅ ALL MODELS TRAINED!")
print("=" * 60)