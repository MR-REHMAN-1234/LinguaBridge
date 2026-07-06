import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("   LINGUABRIDGE - EDA ANALYSIS")
print("=" * 60)

# ─────────────────────────────────────────
# DATASET 1: LANGUAGE DETECTION
# ─────────────────────────────────────────
print("\n📊 DATASET 1: LANGUAGE DETECTION")
print("-" * 40)

df1 = pd.read_csv('data/Language Detection.csv')

print(f"✅ Total Rows:    {df1.shape[0]}")
print(f"✅ Total Columns: {df1.shape[1]}")
print(f"✅ Column Names:  {df1.columns.tolist()}")

print(f"\n🔍 Missing Values:")
print(df1.isnull().sum())

mode_lang = df1['Language'].mode()[0]
print(f"\n📌 Mode (Most Common Language): {mode_lang}")

print(f"\n📊 Har Language Ki Count:")
lang_counts = df1['Language'].value_counts()
print(lang_counts)

df1['text_length'] = df1['Text'].apply(len)
print(f"\n📏 Text Length Analysis:")
print(f"   Mean   (Average Length): {df1['text_length'].mean():.2f}")
print(f"   Median (Middle Length):  {df1['text_length'].median():.2f}")
print(f"   Mode   (Common Length):  {df1['text_length'].mode()[0]}")
print(f"   Std Dev (Spread):        {df1['text_length'].std():.2f}")
print(f"   Min Length:              {df1['text_length'].min()}")
print(f"   Max Length:              {df1['text_length'].max()}")

plt.figure(figsize=(12, 5))
lang_counts.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Language Detection Dataset - Language Distribution', fontsize=14)
plt.xlabel('Language')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('data/lang_distribution.png')
plt.close()
print("\n✅ Graph saved: data/lang_distribution.png")

# ─────────────────────────────────────────
# DATASET 2: SENTIMENT ANALYSIS
# ─────────────────────────────────────────
print("\n\n📊 DATASET 2: SENTIMENT ANALYSIS")
print("-" * 40)

# Pehle 25000 Negative rows
df2_neg = pd.read_csv(
    'data/training.1600000.processed.noemoticon.csv',
    encoding='latin-1',
    header=None,
    nrows=25000
)

# Aakhri 25000 Positive rows
df2_pos = pd.read_csv(
    'data/training.1600000.processed.noemoticon.csv',
    encoding='latin-1',
    header=None,
    skiprows=1600000 - 25000
)

# Dono combine karo
df2 = pd.concat([df2_neg, df2_pos], ignore_index=True)
df2.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

print(f"✅ Loaded Rows:   {df2.shape[0]} (25k Negative + 25k Positive)")
print(f"✅ Total Columns: {df2.shape[1]}")
print(f"✅ Column Names:  {df2.columns.tolist()}")

print(f"\n🔍 Missing Values:")
print(df2.isnull().sum())

print(f"\n📊 Sentiment Distribution:")
sentiment_map = {0: 'Negative', 4: 'Positive'}
df2 = df2[df2['sentiment'].isin([0, 4])]
df2['sentiment_label'] = df2['sentiment'].map(sentiment_map)
print(df2['sentiment_label'].value_counts())

mode_sent = df2['sentiment_label'].mode()[0]
print(f"\n📌 Mode (Most Common Sentiment): {mode_sent}")

df2['text_length'] = df2['text'].apply(len)
print(f"\n📏 Text Length Analysis:")
print(f"   Mean   (Average Length): {df2['text_length'].mean():.2f}")
print(f"   Median (Middle Length):  {df2['text_length'].median():.2f}")
print(f"   Mode   (Common Length):  {df2['text_length'].mode()[0]}")
print(f"   Std Dev (Spread):        {df2['text_length'].std():.2f}")
print(f"   Min Length:              {df2['text_length'].min()}")
print(f"   Max Length:              {df2['text_length'].max()}")

print(f"\n📊 Positive vs Negative Text Length:")
print(df2.groupby('sentiment_label')['text_length'].agg(['mean', 'median', 'std']).round(2))

plt.figure(figsize=(6, 5))
df2['sentiment_label'].value_counts().plot(
    kind='bar',
    color=['red', 'green'],
    edgecolor='black'
)
plt.title('Sentiment Dataset - Distribution', fontsize=14)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('data/sentiment_distribution.png')
plt.close()
print("\n✅ Graph saved: data/sentiment_distribution.png")

print("\n" + "=" * 60)
print("   ✅ EDA COMPLETE!")
print("=" * 60)