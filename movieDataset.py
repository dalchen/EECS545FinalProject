import nltk
import random
import string
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#need to download these packages from nltk initially

#nltk.download('movie_reviews')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
class movieDataset():
	def getTag(tag):
	    if tag.startswith('N'):
	        return 'n'
	    if tag.startswith('J'):
	        return 'a'
	    if tag.startswith('V'):
	        return 'v'
	    if tag.startswith('R'):
	        return 'r'
	    return 'n'

	def clean(review):
	    result=[]
	    for w in review:
	        if w.lower() not in stop:
	            # using pos_tag and lemmatizer for cleaning the review
	            p_tag = pos_tag([w])
	            clean_word = lemmatizer.lemmatize(w,movieDataset.getTag(p_tag[0][1]))
	            result.append(clean_word)
	    return result

if __name__ == '__main__':
	docs=[] 
	for category in movie_reviews.categories():
	    for fileid in movie_reviews.fileids(category):
	        docs.append((movie_reviews.words(fileid),category))        
	random.shuffle(docs)
	stop = stopwords.words('english') + list(string.punctuation)
	lemmatizer = WordNetLemmatizer()

	modified_docs = [(movieDataset.clean(docs), category) for docs, category in docs]
	text_docs = [" ".join(docs) for docs, category in modified_docs]
	category = [category for docs, category in modified_docs]
	x_train, x_test, y_train, y_test = train_test_split(text_docs, category, random_state=0)

	print('Successfully loaded the dataset into train and test set.')
