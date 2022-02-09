import pandas as pd
import numpy as np
import os

class telegram_group:
    def __init__(self,path="bt.json"):
        self.name=path.split(".")[0]
        self.database_path=os.path.join("Data",path)
        self.full_df=pd.read_json(self.database_path)
        messages=self.full_df.loc[:,"messages"].to_list()
        self.messages=pd.DataFrame.from_records(messages)
        self.text_corpus=[re.sub(r'[^\w\s]',' ',word.lower().replace("\n"," ")) for word in self.messages.loc[:,"text"].to_list() if isinstance(word,str)]
        print("__init__ completed")

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