from matplotlib import pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import string
import textmining
import typing



def initData(train, valid):
    colNames = ['unit','name','desc','l1code','l1desc','l2code','l2desc','l3code','l3desc']
    train = pd.read_excel(train)
    valid = pd.read_excel(valid)
    # turn NA/NaN/None into ''
    train = train.fillna('NA')
    valid = valid.fillna('NA')
    train.columns = colNames
    valid.columns = colNames
    trainMat = textmining.TermDocumentMatrix()
    for des in train.desc:
        trainMat.add_doc(str(des))
    validMat = textmining.TermDocumentMatrix()
    for des in valid.desc:
        validMat.add_doc(str(des))
    return trainMat, validMat, train, valid

def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]

def buildMNBClassifier(qasData):
    pipe = Pipeline([
        ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer,
            stop_words=stopwords.words('english') + list(string.punctuation))),
        ('classifier', MultinomialNB(alpha=0.05)),
    ])
    return pipe

def buildSGDClassifier(qasData):
    '''
    🙇
    '''
    sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                        random_state=42, max_iter=5, tol=None)
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=stemming_tokenizer,
            stop_words=stopwords.words('english') + list(string.punctuation))),
        ('clf', sgd),
    ])
    return pipe

def makeClassifier(classifier, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9)
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print(f'Accuracy: {score}')
    return classifier, score

def makeTable(dataDict, fName):
    fig, ax = plt.subplots()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.table(cellText=[list(dataDict.values())],
        colLabels=list(dataDict.keys()))
    plt.show()
    # plt.savefig(fName)