from bs4 import BeautifulSoup
from bs4.element import Comment
from matplotlib import pyplot as plt
from mord.regression_based import OrdinalRidge
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import pickle, os, sys
import requests
import string
import textmining
import typing

# saves me from using capital letters
true = True
false = False

url = 'https://www.newegg.com/global/au-en/Desktop-Graphics-Cards-Desktop-Graphics-Cards/SubCategory/ID-48?Tid=203018&Order=BESTSELLING&PageSize=96'
html = requests.get(url)
soup = BeautifulSoup(html.text, 'html.parser')
links = soup.find_all('a', attrs={'class': 'item-title'})
oldprices = soup.find_all('li', attrs={'class': 'price-was'})
currentprices = soup.find_all('li', attrs={'class': 'price-current'})
savings = soup.find_all('li', attrs={'class': 'price-save'})
titles = []
href_list = []
old_prices = []
prices = []
prices_strong = []
prices_sup = []
discounts = []

for link in links:
    titles.append(link.text)
    href_list.append(link['href'])
for price in list(oldprices):
    prtext = price.find('span', attrs={'class': 'price-currency-label'})
    val = prtext.text if prtext is not None else 'None'
    val = price.text.split('\r')[-1].strip() if price is not None else 'None'
    old_prices.append(val)
for price in currentprices:
    if (price.strong is None or price.sup is None):
        continue
    print(price.strong)
    print(price.sup)
    pr = f'{price.strong.text}{price.sup.text}'
    prices.append(pr)
for discount in savings:
    disc = discount.find('span', attrs={'class': 'price-save-percent'})
    val = disc.text if disc is not None else 'None'
    discounts.append(val)

gpuframe = pd.DataFrame(data=[titles, href_list, old_prices, prices, discounts])
gpuframe = gpuframe.transpose()
gpuframe.columns = ['title', 'link', 'oldprice', 'newprice', 'discount']

def isVisible(elem):
    nonContent = ['style', 'script', 'head', 'meta', 'title', '[document]']
    if elem.parent.name in nonContent: return False
    if isinstance(elem, Comment): return False
    return True

def pageText(responseObj):
    '''
    pass in a requests.Response object
    returns the all the visible text from that page
    '''
    gsoup = BeautifulSoup(responseObj.text, 'html.parser')
    gtext = gsoup.findAll(text=true)
    gvisible = filter(isVisible, gtext)
    return u' '.join(t.strip() for t in gvisible)


print('GPU DATA SUMMARY:')
print(gpuframe.describe().transpose())

ziplist = zip(titles[:10], href_list[:10], prices[:10])
print(f'\nBest selling GPUs: \n----------------------------------------------------------------')
for num, gpu in enumerate(ziplist):
    num += 1
    title, url, price = gpu
    print(f'{num}. {title}\n${price}\n{url}\n----------------------------------------------------------------')

#### Assessment 4 specific code
gpupages = [requests.get(url) for url in href_list]
gpupages = list(map(pageText, gpupages))
gpuframe['pagetext'] = gpupages
gpuframe.to_csv('best_selling_gpu.csv', encoding='utf-8')
# parse components in each page, correlate with popularity
# i.e. try to identify what makes it best selling

'''
with open('gpuframe.pickle', 'wb') as framefile:
    pickle.dump(gpuframe, framefile, pickle.HIGHEST_PROTOCOL)

with open('gpupages.pickle', 'wb') as pagesfile:
    pickle.dump(gpupages, pagesfile, pickle.HIGHEST_PROTOCOL)

with open('gpuframe.pickle', 'rb') as gframe:
    gpuframe = pickle.load(gframe)

with open('gpupages.pickle', 'rb') as gpages:
    gpupages = pickle.load(gpages)
'''

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

def buildLinearRegressor(qasData):
    pipe = Pipeline([
        ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer,
            stop_words=stopwords.words('english') + list(string.punctuation))),
        ('regressor', LinearRegression()),
    ])
    return pipe

def buildOrdinalRegressor(qasData):
    pipe = Pipeline([
        ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer,
            stop_words=stopwords.words('english') + list(string.punctuation))),
        ('regressor', OrdinalRidge()),
    ])
    return pipe

def makeClassifier(classifier, X, y):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=999)
    classifier.fit(Xtrain, ytrain)
    score = classifier.score(Xtest, ytest)
    print(f'Accuracy: {score}')
    return classifier, score


# code below kept for interest, unused because of poor results or unsuitable to task
def buildQDAClassifier(qasData):
    '''
    tfidf = TfidfVectorizer()
    tffit = tfidf.fit_transform(qasData)
    qda = QDA()
    qda.fit(tffit.toarray())
    '''
    qda = QDA()
    pipe = Pipeline([
        # tfidf creates a sparse matrix which qda can't use
        ('tfidf', TfidfVectorizer(tokenizer=stemming_tokenizer,
            stop_words=stopwords.words('english') + list(string.punctuation))),
        # might need to rewrite without pipelines
        #('dense', toarray)
        ('qda', qda),
        ])
    return pipe
