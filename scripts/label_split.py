import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load the fake and true datasets
fake_df = pd.read_csv('D:/projects/ongoing-projs/fake_news_detection/data/balanced_fake_dataset.csv')
true_df = pd.read_csv('D:/projects/ongoing-projs/fake_news_detection/data/True.csv')

# Step 2: Add a label column
fake_df['label'] = 0  # Fake news
true_df['label'] = 1  # True news

# Step 3: Combine both datasets
combined_df = pd.concat([fake_df, true_df], ignore_index=True)

# Step 4: Shuffle the combined dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 5: Split into features and labels
X = combined_df.drop('label', axis=1)
y = combined_df['label']

# Step 6: Split into training and test sets (70-30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Step 7: Reattach labels for saving
train_df = X_train.copy()
train_df['label'] = y_train

test_df = X_test.copy()
test_df['label'] = y_test

# Step 8: Save to CSV
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("✅ train.csv and test.csv files have been created successfully.")
