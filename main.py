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



class utils:
    def __init__(self):
        self.stopwords=["di","a","","da","in","con","su","per","tra","fra",
                    "che","il","lo","la","i","gli","le","un","uno","una",
                    "nan","del","della","e","ha","ho","più","o","dei","hanno","comunque",
                    "alla","cui","è","l","come","anche","al","c","poi","nel"]

class bt:
    def __init__(self):
        self.database_path=os.path.join("Data","bt.json")
        self.full_df=pd.read_json(self.database_path)
        messages=self.full_df.loc[:,"messages"].to_list()
        self.messages=pd.DataFrame.from_records(messages)
        self.text_corpus=self.messages.loc[:,"text"].astype(str).to_list()

    def all_words(self):
        big_str=self.messages.loc[:,"text"].dropna().str.lower().astype(str)
        return re.sub(r'[^\w\s]',' '," ".join(big_str.to_list())).replace("\n"," ").split(" ")

    def word_frequency(self,df=True):
        stopwords=[word for word in utils().stopwords if word in self.all_words()]
        counter=Counter(self.all_words())
        for word in stopwords:
            if word in counter.keys():
                del counter[word]
        if df:
            return pd.DataFrame.from_dict(counter,orient="index").rename(columns={0:"frequency"})
        return counter

    def word_cloud(self):
        text = " ".join([word for word in self.all_words() if not word in utils().stopwords])
        wordcloud = WordCloud().generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")

        wordcloud = WordCloud(max_font_size=40).generate(text)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    def BOW(self):
        corpus=self.text_corpus
        vectorizer=CountVectorizer()#self.text_corpus,input="content")
        X=vectorizer.fit_transform(corpus)
        return vectorizer.get_feature_names_out()


if __name__=="__main__":
    best_timeline=bt()
    #print(best_timeline.BOW())
    best_timeline.word_cloud()
    #temp=best_timeline.word_frequency()
    #print(temp[temp["frequency"]>100].sort_values(by="frequency",ascending=False))
    #print(best_timeline.messages.columns.to_list())
