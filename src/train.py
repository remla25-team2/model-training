import numpy as np
import pandas as pd
import pickle
import joblib
import os 
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from lib_ml.preprocessing import _clean


class SentimentAnalysisModel:
    def __init__(self, dataset_path):
        
        self.dataset = pd.read_csv(dataset_path, delimiter = '\t', quoting = 3)

    def preprocess(self):

        corpus = []
        for i in range(0, 900):
            corpus.append(_clean(self.dataset['Review'][i]))

        return corpus
    
    def transform(self, corpus):

        os.makedirs('bow', exist_ok=True)
        bow_path = 'bow/c1_BoW_Sentiment_Model.pkl'

        cv = CountVectorizer(max_features = 1420)
        X = cv.fit_transform(corpus).toarray()
        y = self.dataset.iloc[:, -1].values
        pickle.dump(cv, open(bow_path, "wb"))

        return X, y

    def splitdata(self, x, y):

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, X_test, y_train, y_test):
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        model_dir = os.path.join('models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'c2_Classifier_Sentiment_Model')
        joblib.dump(classifier, model_path)
        
        
        # y_pred = classifier.predict(X_test)

        # cm = confusion_matrix(y_test, y_pred)

        # accu  racy_score(y_test, y_pred)
        


if __name__ == "__main__":
    sentiment = SentimentAnalysisModel(dataset_path = 'data/a1_RestaurantReviews_HistoricDump.tsv')
    corpus = sentiment.preprocess()
    x, y = sentiment.transform(corpus)

    X_train, X_test, y_train, y_test = sentiment.splitdata(x, y)

    sentiment.train(X_train, X_test, y_train, y_test)






