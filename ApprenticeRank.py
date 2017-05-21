import sys
import json
import time
import copy
import math
import random


import numpy as np
import datetime



from LoadData import *
from Measure import *

def MeasureResult(themap, thendcg):
    string=str(themap)+'\t'
    for ndcg in thendcg:
        string += str(ndcg)+'\t'
    return string


class RL(object):
    def __init__(self, Nfeature, Learningrate, Lenepisode, Resultfile):

        self.Nfeature = Nfeature
        self.Lenepisode = Lenepisode

        self.W=np.random.rand(Nfeature)
        # self.W = np.zeros(Nfeature)
        self.lr = Learningrate

        self.resultfile = open(Resultfile, 'w')
        self.resultfile_w = open(Resultfile + '_W', 'w')

        self.Ntop = 10
        self.memory=[]



    def GenEpisodes(self, Queryids, Data):

        thendcg = np.zeros(self.Ntop)
        themap = 0.0

        nquery = len(Queryids)

        for queryid in Queryids:
            QueryInfo = Data[queryid]
            score = np.dot(QueryInfo['feature'], self.W)
            ndoc = len(score)
            positions = range(ndoc)
            ranklist = np.zeros(ndoc, dtype=np.int32)

            # scoretmp = score.tolist()
            labeltmp = QueryInfo['label'].tolist()

            for position in range(ndoc):
                choice = np.argmax(labeltmp)
                ranklist[position] = positions[choice]

                del labeltmp[choice]
                del positions[choice]

            reward = GetReturn(QueryInfo['label'][ranklist])

            self.memory.append({'queryid': queryid, 'score': score, 'reward': reward, 'ranklist': ranklist})

            rates = QueryInfo['label'][ranklist]
            themap += MAP(rates)
            thendcg += NDCG(self.Ntop, rates)

        themap=themap/nquery
        thendcg = thendcg / nquery

        print  'train: ', themap, thendcg[0], thendcg[2], thendcg[5], thendcg[9],

        self.resultfile.write(MeasureResult(themap, thendcg))


    def UpPolicy(self, Data):
        delta = np.zeros(self.Nfeature)



        for item in self.memory:
            queryid = item['queryid']
            score = item['score']
            reward = item['reward']
            ranklist = item['ranklist']

            QueryInfo = Data[queryid]

            ndoc = len(ranklist)
            delta_query = np.zeros(self.Nfeature)

            lenghth = min(self.Lenepisode, ndoc)

            for position in range(lenghth):
                scorecopy = score[ranklist]
                policy = np.exp(scorecopy) / np.sum(np.exp(scorecopy))

                delta_query += reward[position] * (
                QueryInfo['feature'][ranklist[0]] - np.dot(QueryInfo['feature'][ranklist].transpose(), policy))

                np.delete(ranklist, 0)

            if np.sum(delta_query  * delta_query )>0.0001:
                delta += delta_query  / np.sqrt(np.sum(delta_query  * delta_query ))

        # print delta
        del self.memory[:]

        delta = delta / np.sqrt(np.sum(delta * delta))
        self.W = self.W + self.lr * delta
        self.W = self.W / np.sqrt(np.sum(self.W * self.W ))


        self.resultfile_w.write(self.W)


        # print self.W

    def Eval(self, Data, type):

        thendcg = np.zeros(self.Ntop)
        themap = 0.0

        nquery = len(Data.keys())

        for queryid in Data.keys():
            QueryInfo = Data[queryid]
            score = np.dot(QueryInfo['feature'], self.W)
            rates = QueryInfo['label'][np.argsort(score)[np.arange(len(score) - 1, -1, -1)]]
            themap += MAP(rates)
            thendcg += NDCG(self.Ntop, rates)

            # print NDCG/nquery
        themap = themap / nquery
        thendcg = thendcg / nquery
        self.resultfile.write(MeasureResult(themap, thendcg))

        if type == 'test':
            self.resultfile.write('\n')

        print type, ':  ',themap, thendcg[0], thendcg[2], thendcg[5], thendcg[9]



if __name__ == '__main__':

    dataset = 'MSLR-WEB10K'
    fold = 'Fold1'

    datafile = '/home/zengwei/data/' + dataset + '/' + fold + '/'

    train_data = LoadData(datafile+'train.txt', dataset)
    vali_data  = LoadData(datafile+'vali.txt',  dataset)
    test_data  = LoadData(datafile+'test.txt',  dataset)



    nquery = len(train_data.keys())

    Nfeature=136
    Learningrate=0.01
    Nepisode=100

    Lenepisode=10

    Resultfile = 'ApprenticeRank/Result_'+dataset+'_'+fold+'_'+time.strftime("%m%d", time.localtime())


    learner = RL(Nfeature, Learningrate, Lenepisode, Resultfile)
    learner.Eval(train_data, 'train')
    learner.Eval(vali_data, 'vali')
    learner.Eval(test_data, 'test')
    # np.random.seed(datetime.datetime.now().microsecond)


    for ite in range(10000):
        batch = np.random.randint(nquery,size=Nepisode)

        Queryids=[]
        for i in batch:
            Queryids.append(train_data.keys()[i])

        learner.GenEpisodes(Queryids, train_data)
        learner.UpPolicy(train_data)
        learner.Eval(train_data,'train')
        learner.Eval(vali_data,'vali')
        learner.Eval(test_data,'test')








