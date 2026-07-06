import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle

print("=" * 60)
print("   LINGUABRIDGE - FEATURE ENGINEERING")
print("=" * 60)

# ─────────────────────────────────────────
# STEP 1: DATASETS LOAD KARO
# ─────────────────────────────────────────
print("\n📂 Datasets Load Ho Rahe Hain...")

df1 = pd.read_csv('data/Language_Detection_Clean.csv')
df2 = pd.read_csv('data/Sentiment_Clean.csv')

print(f"✅ Dataset 1 Loaded: {df1.shape[0]} rows")
print(f"✅ Dataset 2 Loaded: {df2.shape[0]} rows")

# ─────────────────────────────────────────
# STEP 2: PREPROCESSING
# ─────────────────────────────────────────
print("\n🧹 STEP 2: PREPROCESSING")
print("-" * 40)

def preprocess(text):
    text = str(text).lower()           # Lowercase
    text = re.sub(r'@\w+', '', text)   # @ mentions hatao
    text = re.sub(r'#\w+', '', text)   # Hashtags hatao
    text = re.sub(r'[^\w\s]', '', text)# Punctuation hatao
    text = re.sub(r'\d+', '', text)    # Numbers hatao
    text = text.strip()                # Extra spaces hatao
    return text

# Dataset 1 pe apply
df1['Text_Clean'] = df1['Text'].apply(preprocess)

# Dataset 2 pe apply
df2['text_clean'] = df2['text'].apply(preprocess)

# Examples dikhao
print("\n📌 Dataset 1 — Preprocessing Example:")
print(f"   Pehle:  {df1['Text'].iloc[0][:50]}...")
print(f"   Baad:   {df1['Text_Clean'].iloc[0][:50]}...")

print("\n📌 Dataset 2 — Preprocessing Example:")
print(f"   Pehle:  {df2['text'].iloc[0][:50]}...")
print(f"   Baad:   {df2['text_clean'].iloc[0][:50]}...")

print("\n✅ Preprocessing Complete!")

# ─────────────────────────────────────────
# STEP 3: LABEL ENCODING
# ─────────────────────────────────────────
print("\n🏷️  STEP 3: LABEL ENCODING")
print("-" * 40)

# Dataset 1 — Language column encode karo
le = LabelEncoder()
df1['Language_encoded'] = le.fit_transform(df1['Language'])

# Classes dikhao
print("\n📌 Language Encoding:")
for i, lang in enumerate(le.classes_):
    print(f"   {lang:15} → {i}")

# Example
print(f"\n📌 Example:")
print(f"   'English' → {le.transform(['English'])[0]}")
print(f"   'French'  → {le.transform(['French'])[0]}")

# Dataset 2 — Already encoded hai
print(f"\n📌 Dataset 2 Sentiment:")
print(f"   0 = Negative, 1 = Positive")
print(f"   Already encoded hai ✅")

# LabelEncoder save karo
with open('data/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("\n✅ Label Encoding Complete!")
print("✅ label_encoder.pkl saved!")

# ─────────────────────────────────────────
# STEP 4: FEATURE EXTRACTION — TF-IDF
# ─────────────────────────────────────────
print("\n🔢 STEP 4: FEATURE EXTRACTION — TF-IDF")
print("-" * 40)

# Dataset 1 — TF-IDF
print("\n📌 Dataset 1 TF-IDF:")
tfidf1 = TfidfVectorizer(max_features=5000)
X1 = tfidf1.fit_transform(df1['Text_Clean'])
y1 = df1['Language_encoded']

print(f"   Input shape:  {df1.shape[0]} rows")
print(f"   Output shape: {X1.shape}")
print(f"   Matlab: {X1.shape[0]} texts, {X1.shape[1]} features")

# Dataset 2 — TF-IDF
print("\n📌 Dataset 2 TF-IDF:")
tfidf2 = TfidfVectorizer(max_features=5000)
X2 = tfidf2.fit_transform(df2['text_clean'])
y2 = df2['sentiment']

print(f"   Input shape:  {df2.shape[0]} rows")
print(f"   Output shape: {X2.shape}")
print(f"   Matlab: {X2.shape[0]} texts, {X2.shape[1]} features")

# TF-IDF Example
print("\n📌 TF-IDF Example — Top 10 words Dataset 1:")
feature_names = tfidf1.get_feature_names_out()
print(f"   {list(feature_names[:10])}")

# TF-IDF save karo
with open('data/tfidf1.pkl', 'wb') as f:
    pickle.dump(tfidf1, f)
with open('data/tfidf2.pkl', 'wb') as f:
    pickle.dump(tfidf2, f)

print("\n✅ TF-IDF Complete!")
print("✅ tfidf1.pkl saved!")
print("✅ tfidf2.pkl saved!")

# ─────────────────────────────────────────
# STEP 5: FEATURE SELECTION
# ─────────────────────────────────────────
print("\n🎯 STEP 5: FEATURE SELECTION")
print("-" * 40)
print(f"\n📌 Dataset 1:")
print(f"   Total possible features: {len(tfidf1.vocabulary_)}")
print(f"   Selected features:       5000")
print(f"   Reason: Top 5000 words kaafi hain language detect karne ke liye")

print(f"\n📌 Dataset 2:")
print(f"   Total possible features: {len(tfidf2.vocabulary_)}")
print(f"   Selected features:       5000")
print(f"   Reason: Top 5000 words sentiment detect karne ke liye")

print("\n✅ Feature Selection Complete!")

# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("   📊 FINAL SUMMARY")
print("=" * 60)

print(f"""
Dataset 1 — Language Detection:
   Original Text    → Qualitative
   After Preprocessing → Clean Text
   After TF-IDF     → {X1.shape[1]} numbers (Quantitative) ✅
   After Label Enc  → Language = numbers ✅

Dataset 2 — Sentiment Analysis:
   Original Text    → Qualitative
   After Preprocessing → Clean Text
   After TF-IDF     → {X2.shape[1]} numbers (Quantitative) ✅
   Sentiment        → Already 0/1 ✅

Saved Files:
   ✅ data/label_encoder.pkl
   ✅ data/tfidf1.pkl
   ✅ data/tfidf2.pkl
""")

print("=" * 60)
print("   ✅ FEATURE ENGINEERING COMPLETE!")
print("=" * 60)