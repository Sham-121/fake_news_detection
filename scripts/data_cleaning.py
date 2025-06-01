import pandas as pd

#--------------------------------first dataset-----------------------------------------------------------------------------
# data = pd.read_csv('D:/projects/ongoing-projs/fake_news_detection/data/balanced_fake_dataset.csv')

# df = data.drop('subject', axis=1, inplace=True)
# df = data.drop('subject', axis=1)

# df.to_csv('newbalanced_fake_dataset.csv', index=False)

#--------------------------------second dataset-----------------------------------------------------------------------------
# data1 = pd.read_csv('D:/projects/ongoing-projs/fake_news_detection/data/Fake_Real_News_Data.csv')

# df = data1.drop(data1.columns[data1.columns.str.contains('unnamed', case=False)], axis=1 )

# df.to_csv('newFake_Real_News_Data.csv', index=False)

#--------------------------------third dataset-----------------------------------------------------------------------------
# data2 = pd.read_csv('D:/projects/ongoing-projs/fake_news_detection/data/news.csv')

# df = data2.drop(data2.columns[data2.columns.str.contains('unnamed', case=False)], axis=1)
# df.to_csv('newNews.csv', index=False)

#--------------------------------fourth dataset-----------------------------------------------------------------------------
# data3 = pd.read_csv('D:/projects/ongoing-projs/fake_news_detection/data/True.csv')

# df = data3.drop('subject', axis=1, inplace=True)
# df = data3.drop('date', axis=1)

# df.to_csv('newTrue.csv', index=False)

#-------------------------------- dataset-----------------------------------------------------------------------------
data4 = pd.read_csv('D:/projects/ongoing-projs/fake_news_detection/data/newbalanced_fake_dataset.csv')

df = data4.drop('date', axis=1)

df.to_csv('newlybalanced_fake_dataset.csv', index=False)