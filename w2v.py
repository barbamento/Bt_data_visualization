import pandas as pd
import gensim
from gensim import corpora,models,utils
from pprint import pprint
import numpy as np
from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count
from gensim.test.utils import datapath
import gensim.downloader as api



class w2v:
    def __init__(self,text):
        self.text=text
        self.create_dict()
        self.create_BOW()
        self.create_corpus()
        #self.tdidf(text)

    def create_dict(self,sep=","):
        if isinstance(self.text,str):
            self.text=self.text.split(sep)
        elif isinstance (self.text,list):
            pass
        else:
            raise ValueError ("text is nor str nor list. This class doesn't support {}".format(type(self.text)))
        self.dictionary=corpora.Dictionary(self.text)
    
    def create_BOW(self):
        mydict=corpora.Dictionary()
        mycorpus=[mydict.doc2bow(doc, allow_update=True) for doc in self.text]
        self.word_counts = [[(mydict[id], count) for id, count in line] for line in mycorpus]
        
    def tdidf(self):
        corpus=[self.dictionary.doc2bow(doc, allow_update=True) for doc in self.text]
        self.tfidf = models.TfidfModel(corpus, smartirs='ntc')

    def create_corpus(self):
        sentences=self.text
        print("begin training model")
        training_loss=10000000
        i=0
        while training_loss>10:
            i+=1
            model = gensim.models.Word2Vec(sentences=sentences,min_count=10,vector_size=50,workers=8,compute_loss=True)
            #print(model.wv["lago"])
            #print("begin evaluation")
            #print(model.wv.evaluate_word_analogies(datapath('questions-words.txt')))
            #print("evaluation completed")
            #training_loss = model.get_latest_training_loss()
            print("iterazione numero : {} , loss = {}".format(i,model.get_latest_training_loss()))
        print("model trained")

class glove:
    def __init__(self):
        pass

if __name__=="__main__":
    pass