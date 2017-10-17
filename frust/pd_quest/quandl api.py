import pandas as pd
import quandl
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

_key = open('quandl_api.txt').read()
quandl.ApiConfig.api_key = 'rWS-9gUcXQiPzziiq1oF'

#df = quandl.get('FMAC/HPI_AK')
#print(df.head())



def state_list():
    fifty_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states ')
    return fifty_states[0][0][1:]
#this gives us a list
#print (fifty_states[0][0])
#this is now a dataframe , a column actually
def grab_init_state_data():
    states = state_list()
    main_df = pd.DataFrame()

    for abbv in states:
        query="FMAC/HPI_" + str(abbv)
        df = quandl.get(query)
        df.rename(columns={'Value': abbv}, inplace=True)
        df[abbv] = (df[abbv] - df[abbv][0]) / df[abbv][0] * 100

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df,lsuffix=abbv)

    print(main_df.head())
    #main_df.to_pickle('new.pickle')
    #return HPI_pd_data
    #an example of piclkle

    pickle_out = open('fifty_states_pct3.pickle', 'wb')
    pickle.dump(main_df,pickle_out)
    pickle_out.close()


def HPI_benchmark():
    df = quandl.get("FMAC/HPI_USA", authtoken="rWS-9gUcXQiPzziiq1oF")
    df.rename(columns={'Value':'United States'}, inplace=True)
    df['United States'] = (df['United States'] - df['United States'][0])\
                          / df['United States'][0] * 100
    return df

def mortgage():
    df = quandl.get("FMAC/30US",trim_start= "1975-01-01", authtoken="rWS-9gUcXQiPzziiq1oF")
    df.rename(columns={'Value': 'm30'}, inplace=True)
    df['m30'] = (df['m30'] - df['m30'][0]) \
                          / df['m30'][0] * 100
    #df = df.resample('D')
    df = df.resample('M').mean()
    return  df


HPI_Bench = HPI_benchmark()
m30 = mortgage()
HPI_data =pd.read_pickle('fifty_states_pct3.pickle')
state_HPI_m30 = HPI_data.join(m30)


print(state_HPI_m30.corr()['m30'].describe( ))

#already pickled
#grab_init_state_data()
#print(data)
#pickle_in = open('fifty_states.pickle', 'rb')

#fig = plt.figure()
#ax1 = plt.subplot2grid((2,1),(0,0))
##ax2 = plt.subplot2grid((2,1),(1,0),sharex=ax1)

#HPI_data =pd.read_pickle('fifty_states_pct3.pickle')
##RResampling
#TX_year =  HPI_data['TX'].resample('A').mean()
#TX_year_OHLC =  HPI_data['TX'].resample('A').ohlc()
#print(TX_year.head())
#print(TX_year_OHLC.head())


#benchmark = HPI_benchmark()


#HPI_data = pickle.load(pickle_in)
#print HPI_data


#TX_year.plot(ax=ax1,label='Yearly TX HPI')
#TX_year_OHLC.plot(ax=ax1,label='Yearly OHLC TX HPI')

#benchmark.plot(ax=ax1, color='k', linewidth=10)

##Ham=ndling missing data
#HPI_data['TX_yr'] = TX_year
#HPI_data.fillna(value=-99999,limit=20, inplace=True)#method=bfill,ffill
#print(HPI_data.isnull().values.sum())

#HPI_data['TX12MA'] = pd.rolling_mean(HPI_data['TX'],12)
#HPI_data['TX12STD'] = pd.rolling_std(HPI_data['TX'],12)

#TX_AK_12corr = HPI_data['TX'].rolling(12).corr(HPI_data['AK'])

#HPI_data[['TX','TX_yr']].plot(ax= ax1, label='Monthly TX HPI')
#HPI_data[['TX','TX12MA']].plot(ax= ax1, label='TX12MA')
#HPI_data.dropna(inplace=True)
#print(HPI_data[['TX','TX12MA']])

#HPI_data['TX12STD'].plot(ax=ax2)
#HPI_data['AK'].plot(ax=ax1,label='AK_HPI')
#HPI_data['TX'].plot(ax=ax1,label='TX_HPI')
#ax1.legend(loc=4)

#TX_AK_12corr.plot(ax=ax2, label='TX_AK_Corr')
#print(TX_AK_12corr)
#plt.legend(loc=4)
#plt.show()


###########correlation

#HPI_State_correlation = HPI_data.corr()
#print(HPI_State_correlation)
#print(HPI_State_correlation.describe())


