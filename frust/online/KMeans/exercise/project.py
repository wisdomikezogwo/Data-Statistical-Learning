import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report

univ = pd.read_csv('College_Data',index_col=0)
print univ.head(), univ.info()

#plt.scatter(univ['Grad.Rate'], univ['Room.Board'],c=univ['Private'],cmap='rainbow')
#plt.show()

sns.set_style('whitegrid')
#sns.lmplot('Room.Board', 'Grad.Rate', data = univ, hue = 'Private', palette='coolwarm'
#          ,size=6, aspect =1 , fit_reg=False)

#sns.lmplot('Outstate', 'F.Undergrad', data = univ, hue = 'Private', palette='coolwarm'
#           ,size=6, aspect =1 , fit_reg=False)

facet = sns.FacetGrid(univ, hue= 'Private', palette='coolwarm', size=6 , aspect=2
)
facet = facet.map(plt.hist, 'Outstate', bins = 20, alpha = 0.7)
#sns.plt.show()

univ['Grad.Rate']['Cazenovia College'] = 100
print(univ[univ['Grad.Rate'] > 100])

#KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(univ.drop('Private', axis=1))

print kmeans.labels_

def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0

univ['Cluster'] = univ['Private'].apply(converter)
print(confusion_matrix(univ['Cluster'],kmeans.labels_))
print(classification_report(univ['Cluster'],kmeans.labels_))
