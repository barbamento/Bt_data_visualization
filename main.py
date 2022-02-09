from pyexpat.errors import messages
import matplotlib
import pandas as pd
import os
import json
import numpy as np
from collections import Counter
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import tqdm
from w2v import w2v

class utils:
    def __init__(self):
        with open("stopwords.txt","r") as f:
            self.stopwords=f.read().split("\n")

class we:
    def __init__(self,paths=["bt.json","dario.json","no_vax.json","polmemes.json","unibo.json"]):
        self.paths=[os.path.join("Data",path) for path in paths]
        dfs=[]
        for path in self.paths:
            dfs+=[pd.read_json(path)]
        df=pd.concat(dfs)
        print(df)
        messages=df.loc[:,"messages"].to_list()
        self.messages=pd.DataFrame.from_records(messages)
        self.text_corpus=[re.sub(r'[^\w\s]',' ',word.lower().replace("\n"," ")) for word in self.messages.loc[:,"text"].to_list() if isinstance(word,str)]
        print("__init__ completed")
        #print(self.text_corpus)
        text=[word.split(" ") for word in self.text_corpus]
        w2v(text)



class bt:
    def __init__(self,path="bt.json"):
        self.name=path.split(".")[0]
        self.database_path=os.path.join("Data",path)
        self.full_df=pd.read_json(self.database_path)
        messages=self.full_df.loc[:,"messages"].to_list()
        self.messages=pd.DataFrame.from_records(messages)
        self.text_corpus=[re.sub(r'[^\w\s]',' ',word.lower().replace("\n"," ")) for word in self.messages.loc[:,"text"].to_list() if isinstance(word,str)]
        print("__init__ completed")

    def all_words(self):
        big_str=self.messages.loc[:,"text"].dropna().str.lower().astype(str)
        return re.sub(r'[^\w\s]',' '," ".join(big_str.to_list())).replace("\n"," ").split(" ")

    def word_frequency(self,df=True):
        counter=Counter(self.all_words())
        for word in set(utils().stopwords).intersection(set(counter.keys())):
            if word in counter.keys():
                del counter[word]
        if df:
            return pd.DataFrame.from_dict(counter,orient="index").rename(columns={0:"frequency"}).sort_values(by="frequency",ascending=False)
        return counter

    def word_cloud(self,how="wc"):
        text=" ".join(self.all_words())
        wordcloud = WordCloud(width=800,height=600,stopwords=utils().stopwords).generate(text)
        plt.figure(figsize=(20,10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(os.path.join("images",str(self.name)+".pdf"))

    def BOW(self):
        corpus=self.text_corpus
        vectorizer=CountVectorizer()#self.text_corpus,input="content")
        X=vectorizer.fit_transform(corpus)
        return vectorizer.get_feature_names_out()

    def create_we(self):
        text=[word.split(" ") for word in self.text_corpus]
        w2v(text)


if __name__=="__main__":
    we()
#    best_timeline=bt()
#    best_timeline.create_we()

