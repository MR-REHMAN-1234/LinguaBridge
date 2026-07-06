import pandas as pd
import numpy as np

print("=" * 60)
print("   LINGUABRIDGE - DATA CLEANING")
print("=" * 60)

# ─────────────────────────────────────────
# DATASET 1: LANGUAGE DETECTION
# ─────────────────────────────────────────
print("\n📊 DATASET 1: LANGUAGE DETECTION")
print("-" * 40)

df1 = pd.read_csv('data/Language Detection.csv')

print(f"Pehle — Rows: {df1.shape[0]}")

# Step 1: Missing Values Check
print(f"\n🔍 Missing Values:")
print(df1.isnull().sum())

# Step 2: Duplicates Check aur Remove
duplicates = df1.duplicated().sum()
print(f"\n🔍 Duplicate Rows: {duplicates}")
df1.drop_duplicates(inplace=True)
print(f"Duplicates hatane ke baad: {df1.shape[0]} rows")

# Step 3: Text Length Column banao
df1['text_length'] = df1['Text'].apply(len)

# Step 4: IQR Method - Outliers Hatao
Q1 = df1['text_length'].quantile(0.25)
Q3 = df1['text_length'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(f"\n📊 IQR Analysis:")
print(f"   Q1:          {Q1}")
print(f"   Q3:          {Q3}")
print(f"   IQR:         {IQR}")
print(f"   Lower Limit: {lower}")
print(f"   Upper Limit: {upper}")

outliers = df1[
    (df1['text_length'] < lower) |
    (df1['text_length'] > upper)
]
print(f"\n⚠️  Outliers found: {len(outliers)} rows")

df1_clean = df1[
    (df1['text_length'] >= lower) &
    (df1['text_length'] <= upper)
]

print(f"\n✅ Cleaning ke baad: {df1_clean.shape[0]} rows")

# Step 5: Clean Dataset Save Karo
df1_clean.to_csv('data/Language_Detection_Clean.csv', index=False)
print("✅ Clean dataset saved: data/Language_Detection_Clean.csv")

print(f"\n📊 Before vs After:")
print(f"   {'':20} {'Pehle':>10} {'Baad':>10}")
print(f"   {'Rows':20} {df1.shape[0]:>10} {df1_clean.shape[0]:>10}")
print(f"   {'Mean':20} {124.06:>10.2f} {df1_clean['text_length'].mean():>10.2f}")
print(f"   {'Median':20} {100.0:>10.2f} {df1_clean['text_length'].median():>10.2f}")
print(f"   {'Std Dev':20} {253.69:>10.2f} {df1_clean['text_length'].std():>10.2f}")
print(f"   {'Max':20} {19088:>10} {df1_clean['text_length'].max():>10}")

print("\n" + "=" * 60)
print("   ✅ DATASET 1 CLEANING COMPLETE!")
print("=" * 60)

# ─────────────────────────────────────────
# DATASET 2: SENTIMENT ANALYSIS
# ─────────────────────────────────────────
print("\n\n📊 DATASET 2: SENTIMENT ANALYSIS")
print("-" * 40)

# Load karo — 25k Negative + 25k Positive
df2_neg = pd.read_csv(
    'data/training.1600000.processed.noemoticon.csv',
    encoding='latin-1',
    header=None,
    nrows=25000
)
df2_pos = pd.read_csv(
    'data/training.1600000.processed.noemoticon.csv',
    encoding='latin-1',
    header=None,
    skiprows=1600000 - 25000
)
df2 = pd.concat([df2_neg, df2_pos], ignore_index=True)
df2.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

print(f"Pehle — Rows: {df2.shape[0]}")

# Step 1: Sirf kaam ke columns rakho
df2 = df2[['sentiment', 'text']]
print(f"Kaam ke Columns: {df2.columns.tolist()}")

# Step 2: Missing Values Check
print(f"\n🔍 Missing Values:")
print(df2.isnull().sum())

# Step 3: Duplicates Check aur Remove
duplicates = df2.duplicated().sum()
print(f"\n🔍 Duplicate Rows: {duplicates}")
df2.drop_duplicates(inplace=True)
print(f"Duplicates hatane ke baad: {df2.shape[0]} rows")

# Step 4: Format Fix — 4 ko 1 mein badlo
df2['sentiment'] = df2['sentiment'].replace(4, 1)
print(f"\n📊 Sentiment Values ab: {df2['sentiment'].unique()}")
print(f"   0 = Negative, 1 = Positive")

# Step 5: Sentiment Distribution check karo
print(f"\n📊 Sentiment Distribution:")
print(df2['sentiment'].value_counts())

# Step 6: IQR Method - Outliers Hatao
df2['text_length'] = df2['text'].apply(len)
Q1 = df2['text_length'].quantile(0.25)
Q3 = df2['text_length'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(f"\n📊 IQR Analysis:")
print(f"   Q1:          {Q1}")
print(f"   Q3:          {Q3}")
print(f"   IQR:         {IQR}")
print(f"   Lower Limit: {lower}")
print(f"   Upper Limit: {upper}")

outliers = df2[
    (df2['text_length'] < lower) |
    (df2['text_length'] > upper)
]
print(f"\n⚠️  Outliers found: {len(outliers)} rows")

df2_clean = df2[
    (df2['text_length'] >= lower) &
    (df2['text_length'] <= upper)
]

print(f"\n✅ Cleaning ke baad: {df2_clean.shape[0]} rows")

# Step 7: Clean Dataset Save Karo
df2_clean.to_csv('data/Sentiment_Clean.csv', index=False)
print("✅ Clean dataset saved: data/Sentiment_Clean.csv")

print(f"\n📊 Before vs After:")
print(f"   {'':20} {'Pehle':>10} {'Baad':>10}")
print(f"   {'Rows':20} {df2.shape[0]:>10} {df2_clean.shape[0]:>10}")
print(f"   {'Mean':20} {74.08:>10.2f} {df2_clean['text_length'].mean():>10.2f}")
print(f"   {'Median':20} {69.0:>10.2f} {df2_clean['text_length'].median():>10.2f}")
print(f"   {'Std Dev':20} {36.27:>10.2f} {df2_clean['text_length'].std():>10.2f}")
print(f"   {'Max':20} {359:>10} {df2_clean['text_length'].max():>10}")

print("\n" + "=" * 60)
print("   ✅ DATASET 2 CLEANING COMPLETE!")
print("=" * 60)

print("\n\n📁 SAVED FILES:")
print("   ✅ data/Language_Detection_Clean.csv")
print("   ✅ data/Sentiment_Clean.csv")
print("\n🎉 DATA CLEANING MUKAMMAL HO GAYI!")