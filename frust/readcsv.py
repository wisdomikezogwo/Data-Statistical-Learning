import pandas as pd

df = pd.read_csv('/home/ikezogwo/Downloads/ZILLOW-Z77006_MLPAH.csv')

print(df.head())

df.set_index('Date', inplace=True)

# df.to_csv('77006csvfile.csv')
print(df)

df = pd.read_csv('77006csvfile.csv', index_col=0)
print(df.head())

df.columns = ['AUSTIN_HPI']
print df.head()

df.to_csv('newcsv.csv')

df.to_html('example1.html')

df = pd.read_csv('newcsv.csv')
print df.head()

df.rename(columns={'AUSTIN_HPI':'77066_HPI'}, inplace=True)

print '                     '
print(df.head())