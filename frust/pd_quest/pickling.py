import pandas as pd
import quandl


def grab_init_state_data():
    states = state_list()
    main_df = pd.DataFrame()

    for abbv in states:
        query="FMAC/HPI_" + str(abbv)
        df = quandl.get(query)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df,lsuffix=abbv)

    print(main_df.head())
    main_df.to_pickle('new.pickle')
    HPI_pd_data = pd.read_pickle('new.pickle')
    return HPI_pd_data




#an example of piclkle

    #pickle_out = open('fifty_states.pickle', 'wb')
    #pickle.dump(main_df,pickle_out)
    #pickle_out.close()
#already pickled
data = grab_init_state_data()
print(data)
#pickle_in = open('fifty_states.pickle', 'rb')
#HPI_data = pickle.load(pickle_in)
#print HPI_data
