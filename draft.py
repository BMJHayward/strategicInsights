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


def buildQDAClassifier(qasData):
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

def lev_distance(str1, str2):
    # calculates levenshtein similarity between 2 strings
    xlen, ylen = len(str1) + 1, len(str2) + 1
    distmatrix = np.zeros((xlen, ylen))
    for x in range(xlen): distmatrix[x,0] = x
    for y in range(ylen): distmatrix[0,y] = y
    for x in range(1, xlen):
        for y in range(1, ylen):
            if str1[x-1] == str2[y-1]:
                distmatrix[x,y] = min(
                    distmatrix[x-1,y] + 1,
                    distmatrix[x-1,y-1],
                    distmatrix[x,y-1] + 1)
            else:
                distmatrix[x,y] = min(
                    distmatrix[x-1,y] + 1,
                    distmatrix[x-1,y-1] + 1,
                    distmatrix[x,y-1] + 1)
    return distmatrix[xlen-1, ylen-1]

def lev_ratio(str1, str2):
    # calculates levenshtein distance as a ratio of the maximum edit distance
    upper_bound = max(len(str1), len(str2))
    return 1 - lev_distance(str1, str2) / upper_bound

def tfidf_cosine_distance(str1, str2):
    # returns a float, a higher number is closer/better result
    vectors = TfidfVectorizer(min_df=1)
    tfidf = vectors.fit_transform([str1, str2])
    return (tfidf*tfidf.T).A[0,1]

def getTopicList(documents: typing.List[str]):
    # perform LDA
    def display_topics(model, feature_names, no_top_words):
        return {
            "Topic {}:".format(topic_idx):" ".join([feature_names[i]
                for i in topic.argsort()[:-no_top_words - 1:-1]])
                    for topic_idx, topic in enumerate(model.components_)
            }
    no_features = 1000
    # LDA can only use raw term counts because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(min_df=0, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
    no_topics = 10
    # Run LDA
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    no_top_words = 10
    return display_topics(lda, tf_feature_names, no_top_words)

vecLevRatio = np.vectorize(lev_ratio)
vecTfidfCosDist = np.vectorize(tfidf_cosine_distance)

def isVisible(elem):
    nonContent = ['style', 'script', 'head', 'meta', 'title', '[document]']
    if elem.parent.name in nonContent: return False
    if isinstance(elem, Comment): return False
    return True

gsoup = BeautifulSoup(gpupages[0].text, 'html.parser')
gtext = gsoup.findAll(text=true)
gvisible = filter(isVisible, gtext)
gpagetext = u' '.join(t.strip() for t in gvisible)
print(gpagetext)

for gpage in gpupages:
    gsoup = BeautifulSoup(gpage.text, 'html.parser')
    gtext = gsoup.findAll(text=true)
    gvisible = filter(isVisible, gtext)
    gpagetext = u' '.join(t.strip() for t in gvisible)