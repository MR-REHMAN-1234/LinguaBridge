import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("   LINGUABRIDGE - SKEWNESS ANALYSIS & FIX")
print("=" * 60)

# ─────────────────────────────────────────
# DATASETS LOAD KARO
# ─────────────────────────────────────────
print("\n📂 Datasets Load Ho Rahe Hain...")
df1 = pd.read_csv('data/Language_Detection_Clean.csv')
df2 = pd.read_csv('data/Sentiment_Clean.csv')
print(f"✅ Dataset 1: {df1.shape[0]} rows")
print(f"✅ Dataset 2: {df2.shape[0]} rows")

# text_length column banao
df1['text_length'] = df1['Text'].apply(len)
df2['text_length'] = df2['text'].apply(len)

# ─────────────────────────────────────────
# DATASET 1 - SKEWNESS CHECK + FIX
# ─────────────────────────────────────────
print("\n📊 DATASET 1 — LANGUAGE DETECTION")
print("-" * 40)

skew1_before = df1['text_length'].skew()
print(f"\n📌 Skewness Before Fix: {skew1_before:.4f}")
print(f"   Mean:   {df1['text_length'].mean():.2f}")
print(f"   Median: {df1['text_length'].median():.2f}")

if skew1_before > 1:
    print("⚠️  Highly Right Skewed!")
elif skew1_before > 0.5:
    print("⚠️  Moderately Right Skewed — Fix Karein!")
elif skew1_before < -1:
    print("⚠️  Highly Left Skewed!")
else:
    print("✅ Normal Range!")

# Log Transformation apply karo
df1['text_length_log'] = np.log1p(df1['text_length'])
skew1_after = df1['text_length_log'].skew()

print(f"\n📌 Log Transformation Apply Ki!")
print(f"   Skewness After Fix: {skew1_after:.4f}")

if abs(skew1_after) < 0.5:
    print("✅ Fixed — Normal Range!")
else:
    print("⚠️  Better hua lekin thoda skewed")

# Graph banao
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df1['text_length'], bins=50,
             color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(df1['text_length'].mean(),
                color='red', linestyle='--',
                label=f"Mean={df1['text_length'].mean():.0f}")
axes[0].axvline(df1['text_length'].median(),
                color='green', linestyle='--',
                label=f"Median={df1['text_length'].median():.0f}")
axes[0].set_title(f'Dataset 1 — Before Fix\nSkewness={skew1_before:.4f}')
axes[0].set_xlabel('Text Length')
axes[0].set_ylabel('Count')
axes[0].legend()

axes[1].hist(df1['text_length_log'], bins=50,
             color='green', edgecolor='black', alpha=0.7)
axes[1].axvline(df1['text_length_log'].mean(),
                color='red', linestyle='--',
                label=f"Mean={df1['text_length_log'].mean():.2f}")
axes[1].axvline(df1['text_length_log'].median(),
                color='orange', linestyle='--',
                label=f"Median={df1['text_length_log'].median():.2f}")
axes[1].set_title(f'Dataset 1 — After Log Fix\nSkewness={skew1_after:.4f}')
axes[1].set_xlabel('Log Text Length')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
plt.savefig('data/skewness_dataset1.png')
plt.close()
print("✅ Graph saved: data/skewness_dataset1.png")

# CSV update karo
df1.to_csv('data/Language_Detection_Clean.csv', index=False)
print("✅ Dataset 1 CSV updated — text_length_log added!")

# ─────────────────────────────────────────
# DATASET 2 - SKEWNESS CHECK + FIX
# ─────────────────────────────────────────
print("\n\n📊 DATASET 2 — SENTIMENT ANALYSIS")
print("-" * 40)

skew2_before = df2['text_length'].skew()
print(f"\n📌 Skewness Before Fix: {skew2_before:.4f}")
print(f"   Mean:   {df2['text_length'].mean():.2f}")
print(f"   Median: {df2['text_length'].median():.2f}")

if skew2_before > 1:
    print("⚠️  Highly Right Skewed!")
elif skew2_before > 0.5:
    print("⚠️  Moderately Right Skewed — Fix Karein!")
elif skew2_before < -1:
    print("⚠️  Highly Left Skewed!")
else:
    print("✅ Normal Range — Fix ki zaroorat nahi!")

# Log Transformation apply karo
df2['text_length_log'] = np.log1p(df2['text_length'])
skew2_after = df2['text_length_log'].skew()

print(f"\n📌 Log Transformation Apply Ki!")
print(f"   Skewness After Fix: {skew2_after:.4f}")

# Graph banao
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df2['text_length'], bins=50,
             color='purple', edgecolor='black', alpha=0.7)
axes[0].axvline(df2['text_length'].mean(),
                color='red', linestyle='--',
                label=f"Mean={df2['text_length'].mean():.0f}")
axes[0].axvline(df2['text_length'].median(),
                color='green', linestyle='--',
                label=f"Median={df2['text_length'].median():.0f}")
axes[0].set_title(f'Dataset 2 — Before Fix\nSkewness={skew2_before:.4f}')
axes[0].set_xlabel('Text Length')
axes[0].set_ylabel('Count')
axes[0].legend()

axes[1].hist(df2['text_length_log'], bins=50,
             color='orange', edgecolor='black', alpha=0.7)
axes[1].axvline(df2['text_length_log'].mean(),
                color='red', linestyle='--',
                label=f"Mean={df2['text_length_log'].mean():.2f}")
axes[1].axvline(df2['text_length_log'].median(),
                color='blue', linestyle='--',
                label=f"Median={df2['text_length_log'].median():.2f}")
axes[1].set_title(f'Dataset 2 — After Log Fix\nSkewness={skew2_after:.4f}')
axes[1].set_xlabel('Log Text Length')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
plt.savefig('data/skewness_dataset2.png')
plt.close()
print("✅ Graph saved: data/skewness_dataset2.png")

# CSV update karo
df2.to_csv('data/Sentiment_Clean.csv', index=False)
print("✅ Dataset 2 CSV updated — text_length_log added!")

# ─────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("   📊 SKEWNESS SUMMARY")
print("=" * 60)
print(f"""
Dataset 1 — Language Detection:
   Skewness Before: {skew1_before:.4f} (Moderately Skewed)
   Skewness After:  {skew1_after:.4f}  (Fixed!)
   Method Used:     Log Transformation

Dataset 2 — Sentiment Analysis:
   Skewness Before: {skew2_before:.4f} (Normal Range)
   Skewness After:  {skew2_after:.4f}  (Applied anyway)
   Method Used:     Log Transformation

CSV Files Updated:
   ✅ data/Language_Detection_Clean.csv
   ✅ data/Sentiment_Clean.csv

Graphs Saved:
   ✅ data/skewness_dataset1.png
   ✅ data/skewness_dataset2.png
""")
print("=" * 60)
print("   ✅ SKEWNESS COMPLETE!")
print("=" * 60)