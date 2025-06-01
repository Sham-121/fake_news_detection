import pandas as pd

# Assuming your data is in a CSV file
df = pd.read_csv('D:/projects/ongoing-projs/fake_news_detection/data/Fake.csv')

# Calculate the number of rows to keep
rows_to_keep = len(df) - 2100

# Remove the last 2100 rows
df = df.head(rows_to_keep)

# Save the modified DataFrame (optional)
df.to_csv('modified_dataset.csv', index=False)