import pandas as pd

df = pd.read_csv('News _dataset\Fake.csv')

n = 2000

df.drop(df.tail(n).index,
        inplace = True)

df.to_csv('balancedFake.csv')

