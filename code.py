import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid', palette='rocket')

df = pd.read_csv(r"C:\Users\victo\Downloads\black_money.csv")

df['Date of Transaction'] = pd.to_datetime(df['Date of Transaction'], errors='coerce')
df = df.dropna(subset=['Date of Transaction'])

print("Shape:", df.shape)
print(df.dtypes)
print(df.describe(include='all'))
print("Missing:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

plt.figure(figsize=(10, 5))
sns.histplot(df['Amount (USD)'], bins=50, kde=True, color='mediumblue')
plt.title("Distribution of Transaction Amounts", fontsize=14)
plt.xlabel("Amount (USD)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 2.5))
sns.boxplot(x='Money Laundering Risk Score', data=df, color='tomato')
plt.title("Risk Score Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
df['Transaction Type'].value_counts().sort_values().plot(kind='barh', color='slateblue')
plt.title("Transaction Type Breakdown")
plt.xlabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
df['Country'].value_counts().head(10).sort_values().plot(kind='barh', color='teal')
plt.title("Top 10 Origin Countries")
plt.xlabel("Count")
plt.tight_layout()
plt.show()

monthly_totals = (
    df.set_index('Date of Transaction')['Amount (USD)']
    .resample('ME')
    .sum()
)
if monthly_totals.nunique() > 1:
    plt.figure(figsize=(12, 5))
    monthly_totals.plot(color='crimson', linewidth=2)
    plt.title("Monthly Total Transaction Volume")
    plt.ylabel("Total Amount (USD)")
    plt.tight_layout()
    plt.show()
else:
    print("Not enough monthly data variation to plot timeline.")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Amount (USD)', y='Money Laundering Risk Score',
                hue='Reported by Authority', alpha=0.7, palette='Set1')
plt.title("Amount vs Risk Score Colored by Authority Reporting")
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 5))
avg_risk = df.groupby('Industry')['Money Laundering Risk Score'].mean().sort_values()
avg_risk.plot(kind='barh', color='darkorange')
plt.title("Average Risk Score by Industry")
plt.xlabel("Average Score")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 3.5))
df['Shell Companies Involved'].value_counts().plot(kind='bar', color='purple')
plt.title("Shell Company Involvement")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
df['Tax Haven Country'].value_counts().head(10).sort_values().plot(kind='barh', color='indigo')
plt.title("Top 10 Tax Haven Countries in Transactions")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Numeric Features")
plt.tight_layout()
plt.show()
