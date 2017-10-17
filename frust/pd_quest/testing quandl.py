import pandas as pd
import quandl
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')


quandl.ApiConfig.api_key = 'rWS-9gUcXQiPzziiq1oF'


def state_list():
    fifty_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states ')
    return fifty_states[0][0][1:]

def grab_init_state_data():

    states = state_list()
    main_df = pd.DataFrame()

    for abbv in states:
        query="FMAC/HPI_" + str(abbv)
        df = quandl.get(query)
        df.rename(columns={'Value': abbv}, inplace=True)
        df[abbv] = (df[abbv]-df[abbv][0]) / df[abbv][0]*100

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df,lsuffix=abbv)


    pickle_out = open('fifty_states_pct2.pickle', 'wb')
    pickle.dump(main_df,pickle_out)
    pickle_out.close()


#grab_init_state_data()

HPI_data =pd.read_pickle('fifty_states_pct3.pickle')

HPI_data.plot()
plt.legend().remove()
plt.show()
