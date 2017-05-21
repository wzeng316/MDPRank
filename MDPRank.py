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


class RL(object):
    def __init__(self, Nfeature, Learningrate, Nepisode, Resultfile):

        self.Nfeature = Nfeature

        self.W=np.random.rand(Nfeature)
        # self.W = np.zeros(Nfeature)
        self.lr = Learningrate

        self.resultfile = open(Resultfile, 'w')
        self.resultfile_w = open(Resultfile + '_w', 'w')

        self.Ntop = 10
        self.memory=[]

    def GenAnEpisode(self, Queryid, Data):

        QueryInfo=Data[Queryid]

        score = np.dot(QueryInfo['feature'], self.W)
        ndoc = len(score)

        positions = range(ndoc)
        ranklist = np.zeros(ndoc, dtype=np.int32)
        for position in range(ndoc):

            policy = np.exp(score) / np.sum(np.exp(score))
            choice = np.random.choice(len(policy), 1, p=policy)[0]
            ranklist[position] = positions[choice]

            del score[choice]
            del positions[choice]

        rates = QueryInfo['label'][ranklist]
        # reward = GetReturn(query['label'][ranklist])
        # self.memory.append({'queryid': Queryid, 'score': score, 'reward': reward, 'ranklist': ranklist})

        themap = MAP(rates)
        thendcg = NDCG(self.Ntop, rates)

        return themap, thendcg

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

            scoretmp= score.tolist()

            for position in range(ndoc):
                policy = np.exp(scoretmp) / np.sum(np.exp(scoretmp))
                choice = np.random.choice(len(policy), 1, p=policy)[0]
                ranklist[position] = positions[choice]

                del scoretmp[choice]
                del positions[choice]

            reward = GetReturn(QueryInfo['label'][ranklist])

            self.memory.append({'queryid': queryid, 'score': score, 'reward': reward, 'ranklist': ranklist})

            rates = QueryInfo['label'][ranklist]
            themap += MAP(rates)
            thendcg += NDCG(self.Ntop, rates)

        themap=themap/nquery
        thendcg = thendcg / nquery

        print  'train: ', themap, thendcg[0], thendcg[2], thendcg[5], thendcg[9],


    def UpPolicy(self, Data):
        delta = np.zeros(self.Nfeature)



        for item in self.memory:
            queryid = item['queryid']
            score = item['score']
            reward = item['reward']
            ranklist = item['ranklist']

            QueryInfo = Data[queryid]

            ndoc = len(ranklist)
            for position in range(ndoc):
                scorecopy = score[ranklist]
                policy = np.exp(scorecopy) / np.sum(np.exp(scorecopy))

                delta = delta + reward[position] * (
                QueryInfo['feature'][ranklist[0]] - np.dot(QueryInfo['feature'][ranklist].transpose(), policy))

                np.delete(ranklist, 0)

        # print delta
        del self.memory[:]

        delta = delta / np.sqrt(np.sum(delta * delta))
        self.W = self.W + self.lr * delta
        self.W = self.W / np.sqrt(np.sum(self.W * self.W ))

        # print self.W

    def Eval(self, Data):

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

        print 'test:  ',themap, thendcg[0], thendcg[2], thendcg[5], thendcg[9]



if __name__ == '__main__':

    dataset = 'MSLR-WEB10K'
    fold = 'Fold1'

    datafile = '/home/zengwei/data/' + dataset + '/' + fold + '/'

    Filename=datafile+'test.txt'
    # Filename='data'
    test_data = LoadData(Filename, dataset)
    nquery = len(test_data.keys())

    Nfeature=136
    Learningrate=0.001
    Nepisode=10
    Resultfile='aaaa'

    learner = RL(Nfeature, Learningrate, Nepisode, Resultfile)
    learner.Eval(test_data)

    # np.random.seed(datetime.datetime.now().microsecond)


    for ite in range(100):
        batch = np.random.randint(nquery,size=Nepisode)

        Queryids=[]
        for i in batch:
            Queryids.append(test_data.keys()[i])

        learner.GenEpisodes(Queryids, test_data)
        learner.UpPolicy(test_data)
        learner.Eval(test_data)








