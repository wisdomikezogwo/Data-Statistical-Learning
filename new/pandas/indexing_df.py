import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import  numpy as np
style.use('ggplot')

web_stat = {'Day':[1,2,3,4,5,6],
            'Visitors':[43,53,34,45,64,34],
            'Bounce_rate':[65,72,62,64,54,66]}

#to convert to a dataframe

df = pd.DataFrame(web_stat)
print(df)
print(df.head())
print(df.tail())
print(df.tail(3))

#to index your dataframe

#df = df.set_index('Day')


#df.set_index('Day',inplace=True)
print(df)
#to reference one or multiple columns
print(df['Visitors'])
print(df.Bounce_rate)

print(df[['Visitors', 'Day']])

#to convert a column  to a list
print(df.Visitors.tolist())


print(np.array(df[['Visitors', 'Day']]))

df1 = pd.DataFrame(np.array(df[['Visitors', 'Day']]))
print(df1)