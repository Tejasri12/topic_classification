import pandas as pd
import numpy as np
from collections import Counter
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import f1_score, classification_report

INPUR_DIR = 'data/training/'

#load data

def load_data(file):
    data, labels = [], []

    with open(''.join([INPUR_DIR, '{}.csv']).format(file)) as f:
        for row in csv.DictReader(f):
            data.append(item['text'])
            labels.append(item['category'])

    return text, targets

X_train, y_train = load_data('train')
X_test, y_test = load_data('test')

from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline

lda = Pipeline([('CV', CountVectorizer(strip_accents='unicode', stop_words='english')),
                ('LDA',LatentDirichletAllocation(n_topics=19, max_iter=80, learning_method='online', 
                                                 learning_offset=50.,doc_topic_prior=.1, topic_word_prior=.01,
                                                 random_state=0)), ])

features =  FeatureUnion([ ('LDA', lda), 
                       ('TFIDF', TfidfVectorizer(strip_accents='unicode', min_df=5, ngram_range=(1, 2)))])

clf = make_pipeline(features, LogisticRegression(C=1.0, penalty='l2', random_state=100), ).fit(X_train, y_train)
pred = lr_model.predict(X_test)

print(classification_report(Y_test, pred))

from nltk import stem
import nltk
from nltk.corpus import stopwords

stemmer = stem.PorterStemmer()

def stem_it(sentence):
     
    stemmed_sequence = [stemmer.stem(word) for word in nltk.word_tokenize(sentence)
                        if word not in stopwords.words('english')]
    return ' '.join(stemmed_sequence)

train_stm = [stem_it(sent) for sent in text_train]
test_stm = [stem_it(sent) for sent in text_test]

#adding LDA features 
lda = Pipeline([('CV', CountVectorizer(strip_accents='unicode', stop_words='english')),
                ('LDA',LatentDirichletAllocation(n_topics=19, max_iter=80, learning_method='online', 
                                                 learning_offset=50.,doc_topic_prior=.1, topic_word_prior=.01,
                                                 random_state=0)), ])

features =  FeatureUnion([ ('LDA', lda), 
                       ('TFIDF', TfidfVectorizer(strip_accents='unicode', min_df=5, ngram_range=(1, 2)))])

final_clf = make_pipeline(features, 
                               xgb.XGBClassifier(max_depth=12, n_estimators=90, learning_rate=0.1, colsample_bytree=.7, 
                                                 gamma=.01, reg_alpha=4, objective='multi:softmax')
                              ).fit(train_stm, y_train) 

xgb_pred = final_clf.predict(stemmed_test)

xgb_f1 = f1_score(test_stm, xgb_pred, average='macro')

print(classification_report(y_test, xgd_pred))

print("f1 score for xgb " , stem_xgb_macro_f1)



