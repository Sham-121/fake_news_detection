import pandas as pd
from sklearn.model_selection import train_test_split

# Load datasets with potential existing labels
df1 = pd.read_csv('D:/projects/ongoing-projs/fake_news_detection/data/newFake_Real_News_Data.csv')
df2 = pd.read_csv('D:/projects/ongoing-projs/fake_news_detection/data/newNews.csv')

# Normalize existing label values (if labeled as 'FAKE'/'REAL')
for df in [df1, df2]:
    if 'label' in df.columns:
        df['label'] = df['label'].replace({'FAKE': 0, 'REAL': 1})
    else:
        print("⚠️ 'label' column not found in one of the datasets")

# Load separate fake and true datasets (no pre-existing labels)
fake_df = pd.read_csv('D:/projects/ongoing-projs/fake_news_detection/data/newlybalanced_fake_dataset.csv')
true_df = pd.read_csv('D:/projects/ongoing-projs/fake_news_detection/data/newTrue.csv')

# Add label columns explicitly
fake_df['label'] = 0  # Fake news
true_df['label'] = 1  # Real news

# Combine all datasets
combined_df = pd.concat([fake_df, true_df, df1, df2], ignore_index=True)

# Clean up label values again to ensure consistency
combined_df['label'] = combined_df['label'].replace({'FAKE': 0, 'REAL': 1})
combined_df['label'] = combined_df['label'].astype(int)  # Force all labels to be int

# Shuffle the combined dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into features and labels
X = combined_df.drop('label', axis=1)
y = combined_df['label']

# Train-test split (80-20) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Reattach labels for saving
train_df = X_train.copy()
train_df['label'] = y_train

test_df = X_test.copy()
test_df['label'] = y_test

# Save to CSV
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("✅ train.csv and test.csv files have been created successfully.")
