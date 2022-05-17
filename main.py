import pandas as pd
import seaborn as sns
from attributes_select import selection
from classification import classification
from data_cleaning import remove_repeat, data_visual, outliers, data_scale
sns.set_style('darkgrid')
RED = '#D10000'
BLUE = '#0082FF'

data = pd.read_csv('/Users/49241/PycharmProjects/ECE677Project2/loldata.csv')
data.head()
data = remove_repeat(data)
data_visual(data)
# data.info()
data_tr = outliers(data)
y = data_tr.blueWins
# y = data.blueWins
#data_final = selection(data_tr, y)
#data_final.info()

#print(data_final.shape, y.shape)
#data_final.head()
#table = classification(data_final, y)
# table = classification(data, y)
table = classification(data_tr, y)
print(table)
