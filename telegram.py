import datetime
from tkinter import Y
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from collections import Counter
import tqdm
import re
import spacy
import json
from spacy import displacy
import plotly.express as px
import plotly.graph_objects as go

class interactive_graph:
    def __init__(self,df):
        self.df=df
        #print(self.df)
        self.frequency()

    def frequency(self,
            x="date",y="extended_text",last_days=90,show_all_days=True):
        df=self.df
        df.loc[:,x]=df.loc[:,x].str[:10]
        if show_all_days:
            days=np.unique(df.loc[:,x])
            all_dates_7=[days[i:i+7] for i in range(len(days)-6)]
            all_dates_30=[days[i:i+30] for i in range(len(days)-29)]
            messages_last_7=[len(df[df[x].isin(dates)].loc[:x])/7 for dates in all_dates_7]
            messages_last_30=[len(df[df[x].isin(dates)].loc[:x])/30 for dates in all_dates_30]
        else:
            last_30_days=[str(datetime.date.today()-datetime.timedelta(days=i)) for i in range(1,last_days+1)]
            df=df[df.loc[:,x].isin(last_30_days)]
        layout=go.Layout(title="number of messages sent on Timeline Observatory",
            xaxis=dict(title="dates"),
            yaxis=dict(title="ammount of messages"))
        fig=go.Figure(layout=layout)
        fig.add_trace(go.Bar(y=df.loc[:,x].value_counts().to_list(),x=df.loc[:,x].value_counts().index.to_list(),name="daily messages"))
        if show_all_days:
            fig.add_trace(go.Scatter(x=days[7:],y=messages_last_7,name="mean of the last 7 days"))
            fig.add_trace(go.Scatter(x=days[30:],y=messages_last_30,name="mean of the last 30 days"))
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                            label="1m",
                            step="month",
                            stepmode="backward"),
                        dict(count=6,
                            label="6m",
                            step="month",
                            stepmode="backward"),
                        dict(count=1,
                            label="1y",
                            step="year",
                            stepmode="backward"),
                        dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)
        fig.write_html("temp.html")


class telegram_group:
    '''
    This class is used to analyze datas from the downloaded json of 
    
    '''
    def __init__(self,path="bt.json"): #riguarda l'implementazione per i we
        #self.we=spacy.load('it_vectors_wiki_lg')
        #self.we=spacy.load("it_core_news_sm")
        self.name=path.split(".")[0]
        self.database_path=os.path.join("Data",path)
        self.full_df=pd.read_json(self.database_path)
        messages=self.full_df.loc[:,"messages"].to_list()
        self.messages=pd.DataFrame.from_records(messages)
        with open("stopwords.txt","r") as f:
            self.stopwords=f.read().split("\n")
        full_sentences=[self.preprocess(word) for word in self.messages.loc[:,"text"]]# if not self.preprocess(word) in [""," "]]
        self.messages.loc[:,"extended_text"]=full_sentences
        self.messages=self.messages[~self.messages["extended_text"].isin([""," "])]
        self.full_sentences=self.messages.loc[:,"extended_text"].to_list()#vedi se serve
        print("__init__ completed")
    
    def preprocess(self,word,
            full_sentence=True): 
        if isinstance(word,str):
            text=word
        elif isinstance(word,list):
            word=word[0]
            if isinstance(word,str):
                text=word
            elif isinstance(word,dict):
                if word["type"]in ["bold","italic","text_link","strikethrough","phone","underline","pre","spoiler"]:
                    text=word["text"]
                elif word["type"] in ["bot_command","mention","link","code","mention_name","email","cashtag"]:
                    text=" "
                elif word["type"] in ["hashtag"]:
                    text=word["text"].replace("#","")
                else:
                    raise ValueError("{} is not implemented in telegram_group.preprocess\nexample : {}".format(word["type"],word))
        elif isinstance(word,float) or isinstance(word,int):
            text=str(word)
        else:
            raise ValueError("{} is not implemented in telegram_group.preprocess\nexample : {}".format(type(word),word))
        text=re.sub(r'[^\w\s]',' ',text.lower().replace("\n"," "))
        splitted_text=[temp for temp in text.split(" ") if not temp in self.stopwords]
        if full_sentence:
            return " ".join(splitted_text)
        else:
            return splitted_text

    def word_frequency(self,df=True):
        counter=Counter([word.split(" ") for word in self.full_sentences])
        for word in set(self.stopwords).intersection(set(counter.keys())):
            if word in counter.keys():
                del counter[word]
        if df:
            return pd.DataFrame.from_dict(counter,orient="index").rename(columns={0:"frequency"}).sort_values(by="frequency",ascending=False)
        return counter

    def word_cloud(self):
        #text=self.full_sentences
        #print(text)
        wordcloud = WordCloud(width=800,height=600,stopwords=self.stopwords).generate(" ".join(self.full_sentences))
        plt.figure(figsize=(20,10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(os.path.join("images",str(self.name)+".pdf"))

    def to_interactive_graph(self):
        return interactive_graph(self.messages)

if __name__=="__main__":
    bt=telegram_group().to_interactive_graph()
