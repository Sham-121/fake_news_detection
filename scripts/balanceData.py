import pandas as pd

true_df = pd.read_csv('News _dataset\True.csv')
fake_df = pd.read_csv('balancedFake.csv')

#1 for TRUE and 0 for FALSE
true_df['label'] = 1
fake_df['label'] = 0

#merging data
combined_df = pd.concat([fake_df, true_df], ignore_index=True)

#shuffling data
shuffled_df = combined_df.sample(frac=1, random_state=42)
shuffled_df = shuffled_df.reset_index(drop=True)

#converting to csv
shuffled_df.to_csv('labeled_dataset.csv', index=False)
