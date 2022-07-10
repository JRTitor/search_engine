import math
from matplotlib.pyplot import title
import pandas as pd
from tkinter import W
import re
import numpy as np
import pickle
from nltk import wordnet, pos_tag
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import math

sw_eng = set(stopwords.words('english'))

data_size = 1186619# размер документов, в def build_index() изменяется в соотвествии с размером 


class Document:
    def __init__(self, title, text):
        # можете здесь какие-нибудь свои поля подобавлять
        self.title = title
        self.text = text
    
    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.text + ' ...']

def load_obj(name):
    #открываем .pickle
    with open('p/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

index = []
to_index = load_obj('to_index.pickle')# открываем индекс для поиска

def build_index():
    # считывает сырые данные и строит индекс

    df_quotes_big = load_obj('data.pickle') #открываем базу данных и запоняем index
    rows = df_quotes_big[df_quotes_big.columns[0]].count()
    data_size = rows

    for i in range(rows):
        found = df_quotes_big.iloc[i]
        index.append(Document(found.iloc[1], found.iloc[0]))

def get_wordnet_pos(treebank_tag):
    my_switch = {
        'J': wordnet.wordnet.ADJ,
        'V': wordnet.wordnet.VERB,
        'N': wordnet.wordnet.NOUN,
        'R': wordnet.wordnet.ADV,
    }
    for key, item in my_switch.items():
        if treebank_tag.startswith(key):
            return item
    return wordnet.wordnet.NOUN      
         
def my_lemmatizer(sent):
    lemmatizer = WordNetLemmatizer()
    tokenized_sent = sent.split()
    pos_tagged = [(word, get_wordnet_pos(tag))
                 for word, tag in pos_tag(tokenized_sent)]
    return ' '.join([lemmatizer.lemmatize(word, tag)
                    for word, tag in pos_tagged])

def score(query, document):
    # возвращает какой-то скор для пары запрос-документ
    # больше -- релевантнее

    if document == np.nan:
        return 0
        
    query = np.array(my_lemmatizer(re.sub( r'[^\w\s]+', '', query.lower())).split())
    doc = document.title + ' ' + document.text
    doc = np.array(my_lemmatizer(re.sub( r'[^\w\s]+', '', doc.lower())).split())

    
    cnt = 0
    for word in query:
        if not (word in sw_eng) and (word in to_index):
            wordindoc = (doc == word).sum()
            cnt += (wordindoc / doc.size ) * math.log((data_size - len(to_index[word]) + 0.5) / (0.5 + len(to_index[word]))) #tf idf

    return cnt

def retrieve(query):
    # возвращает начальный список релевантных документов

    candidates = []
    ids = set()
    query = np.array(my_lemmatizer(re.sub( r'[^\w\s]+', '', query.lower())).split())
    for word in query:
        if (not (word in sw_eng)) and (word in to_index):           # мы хотим, чтобы все "важные" слова были в запросе
            if len(ids) > 0:
                ids = ids & to_index[word]
            else:
                ids = to_index[word]

    for el in ids:
        candidates.append(index[el])

    out = {}
    for doc in candidates:                # присваемваем наибольший скор коротким документам, предпологая, что они релевантнее длинных
        cnt = 5000 / len(doc.text)
        out[cnt] = doc
    
    out = sorted(out.items())
    out =  [i[1] for i in out]

    return out[-50:]    # возвращаем 50 самых коротких документов