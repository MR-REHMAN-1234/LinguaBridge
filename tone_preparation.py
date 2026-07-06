import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle

print("=" * 60)
print("   LINGUABRIDGE - TONE DATASET PREPARATION")
print("=" * 60)

# ─────────────────────────────────────────
# STEP 1: LOAD
# ─────────────────────────────────────────
print("\n📂 Dataset Load Ho Raha Hai...")
df = pd.read_csv('data/tone_dataset.csv')
print(f"✅ Rows: {df.shape[0]}")
print(f"✅ Columns: {df.columns.tolist()}")

# ─────────────────────────────────────────
# STEP 2: EDA
# ─────────────────────────────────────────
print("\n📊 STEP 2: EDA")
print("-" * 40)

print(f"\n📌 Basic Info:")
print(f"   Total Rows:    {df.shape[0]}")
print(f"   Total Columns: {df.shape[1]}")

print(f"\n📌 Missing Values:")
print(df.isnull().sum())

print(f"\n📌 Label Distribution:")
print(df['label'].value_counts())

# Text length
df['text_length'] = df['sentence'].apply(len)

print(f"\n📌 Text Length Analysis:")
print(f"   Mean:   {df['text_length'].mean():.2f}")
print(f"   Median: {df['text_length'].median():.2f}")
print(f"   Mode:   {df['text_length'].mode()[0]}")
print(f"   Std Dev:{df['text_length'].std():.2f}")
print(f"   Min:    {df['text_length'].min()}")
print(f"   Max:    {df['text_length'].max()}")

# Formal vs Informal length comparison
print(f"\n📌 Formal vs Informal Text Length:")
print(df.groupby('label')['text_length'].agg(['mean','median','std']).round(2))

# Graph
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df['label'].value_counts().plot(
    kind='bar', ax=axes[0],
    color=['steelblue','orange'],
    edgecolor='black'
)
axes[0].set_title('Tone Distribution')
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Count')
axes[0].tick_params(rotation=0)

df.groupby('label')['text_length'].mean().plot(
    kind='bar', ax=axes[1],
    color=['steelblue','orange'],
    edgecolor='black'
)
axes[1].set_title('Average Text Length by Tone')
axes[1].set_xlabel('Label')
axes[1].set_ylabel('Avg Length')
axes[1].tick_params(rotation=0)

plt.tight_layout()
plt.savefig('data/tone_eda.png')
plt.close()
print("\n✅ EDA Graph saved: data/tone_eda.png")

# ─────────────────────────────────────────
# STEP 3: CLEANING
# ─────────────────────────────────────────
print("\n🧹 STEP 3: CLEANING")
print("-" * 40)

# Duplicates
dups = df.duplicated().sum()
print(f"\n📌 Duplicates: {dups}")
df.drop_duplicates(inplace=True)
print(f"✅ Rows after removing duplicates: {df.shape[0]}")

# Missing values
print(f"\n📌 Missing Values After Clean:")
print(df.isnull().sum())

# ─────────────────────────────────────────
# STEP 4: SKEWNESS
# ─────────────────────────────────────────
print("\n📈 STEP 4: SKEWNESS")
print("-" * 40)

skew = df['text_length'].skew()
print(f"\n📌 Skewness: {skew:.4f}")

if skew > 1:
    print("⚠️  Highly Right Skewed!")
elif skew > 0.5:
    print("⚠️  Moderately Skewed — Fix Karein!")
elif skew < -1:
    print("⚠️  Highly Left Skewed!")
else:
    print("✅ Normal Range!")

# Log Transform
df['text_length_log'] = np.log1p(df['text_length'])
skew_after = df['text_length_log'].skew()
print(f"📌 After Log Fix: {skew_after:.4f}")

# Skewness Graph
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df['text_length'], bins=20,
             color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(df['text_length'].mean(),
                color='red', linestyle='--',
                label=f"Mean={df['text_length'].mean():.0f}")
axes[0].axvline(df['text_length'].median(),
                color='green', linestyle='--',
                label=f"Median={df['text_length'].median():.0f}")
axes[0].set_title(f'Before Fix — Skewness={skew:.2f}')
axes[0].legend()

axes[1].hist(df['text_length_log'], bins=20,
             color='green', edgecolor='black', alpha=0.7)
axes[1].axvline(df['text_length_log'].mean(),
                color='red', linestyle='--',
                label=f"Mean={df['text_length_log'].mean():.2f}")
axes[1].axvline(df['text_length_log'].median(),
                color='orange', linestyle='--',
                label=f"Median={df['text_length_log'].median():.2f}")
axes[1].set_title(f'After Log Fix — Skewness={skew_after:.2f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('data/tone_skewness.png')
plt.close()
print("✅ Skewness Graph saved: data/tone_skewness.png")

# ─────────────────────────────────────────
# STEP 5: PREPROCESSING
# ─────────────────────────────────────────
print("\n🧹 STEP 5: PREPROCESSING")
print("-" * 40)

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

df['sentence_clean'] = df['sentence'].apply(preprocess)

print(f"\n📌 Example:")
print(f"   Pehle: {df['sentence'].iloc[0]}")
print(f"   Baad:  {df['sentence_clean'].iloc[0]}")

print("\n✅ Preprocessing Complete!")

# ─────────────────────────────────────────
# STEP 6: LABEL ENCODING
# ─────────────────────────────────────────
print("\n🏷️  STEP 6: LABEL ENCODING")
print("-" * 40)

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

print(f"\n📌 Label Encoding:")
for i, label in enumerate(le.classes_):
    print(f"   {label:10} → {i}")

# Save encoder
with open('data/tone_label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("\n✅ Label Encoder saved: data/tone_label_encoder.pkl")

# ─────────────────────────────────────────
# STEP 7: TF-IDF + FEATURE SELECTION
# ─────────────────────────────────────────
print("\n🔢 STEP 7: TF-IDF + FEATURE SELECTION")
print("-" * 40)

tfidf = TfidfVectorizer(max_features=500)
X = tfidf.fit_transform(df['sentence_clean'])
y = df['label_encoded']

print(f"\n📌 TF-IDF Results:")
print(f"   Input:  {df.shape[0]} rows")
print(f"   Output: {X.shape} matrix")
print(f"   max_features=500 (Dataset chhota hai)")

# Save TF-IDF
with open('data/tfidf_tone.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("\n✅ TF-IDF saved: data/tfidf_tone.pkl")

# ─────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("   📊 FINAL SUMMARY")
print("=" * 60)
print(f"""
Dataset:
   Total Rows:    {df.shape[0]}
   Formal:        {df[df['label']=='formal'].shape[0]}
   Informal:      {df[df['label']=='informal'].shape[0]}

EDA:
   Mean Length:   {df['text_length'].mean():.2f}
   Skewness:      {skew:.4f} → {skew_after:.4f} (Fixed)

Cleaning:
   Duplicates:    {dups}
   Missing:       0

Feature Engineering:
   TF-IDF Shape:  {X.shape}
   Labels:        formal=0, informal=1

Saved Files:
   ✅ data/tone_label_encoder.pkl
   ✅ data/tfidf_tone.pkl
   ✅ data/tone_eda.png
   ✅ data/tone_skewness.png
""")
print("=" * 60)
print("   ✅ TONE PREPARATION COMPLETE!")
print("=" * 60)