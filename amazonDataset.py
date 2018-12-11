#code from http://jmcauley.ucsd.edu/data/amazon/
import json
import gzip
import pandas as pd
import gzip
from sklearn.model_selection import train_test_split, cross_val_score

class amazonDataset():
	def parse(path):
	    g = gzip.open(path, 'r')
	    for l in g:
	        yield eval(l)
	    
	def getDF(path):
	    i = 0
	    df = {}
	    for d in amazonDataset.parse(path):
	        df[i] = d
	        i += 1
	    ''' Return python dataframe of the data '''
	    return pd.DataFrame.from_dict(df, orient='index')

if __name__ == '__main__':
	#load the amazon dataset
	amazon = amazonDataset.getDF('reviews_Musical_Instruments_5.json.gz')
	#split train and test data #overall refers to the ratings
	x_train, x_test, y_train, y_test = train_test_split(amazon.reviewText, amazon.overall, random_state=0)
	print('Successfully loaded the dataset into train and test set.')
