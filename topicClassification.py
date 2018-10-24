import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

data = pd.read_csv("categories/data")
print(data.info())

cntvect = CountVectorizer()
data_cv = cntvect.fit_transform(data[:,0:-1])
tfidf = TfidfTransformer()
tfidf = tfidf.fit_transform(data_cv)
clf = MultinomialNB().fit(tfidf, data['category'])
nb_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
nb_clf = nb_clf.fit(data[:,0:-1], data['category'])
nb_pred = nb_clf.predict(data[:,0:-1])
print(np.mean(nb_pred == data['category']))
        
import numpy as np
import nltk
nltk.download()
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)
class Stemit(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
stem_vect = Stemit(stop_words='english')

from sklearn.linear_model import SGDClassifier
svm_clf = Pipeline([('vect', stem_vect),('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5,random_state=42)),])
svm_clf = svm_clf.fit(data[:,0:-1], data['category'])
svm_pred = svm_clf.predict(data[:,0:-1])
print(np.mean(svm_pred == data['category']))

from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3),}
clf = GridSearchCV(svm_clf, parameters, n_jobs=-1)
clf = clf.fit(data[:,0:-1], data['category'])

print(clf.best_score_)
print(clf.best_params_)
