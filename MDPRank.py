import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import yaml
import tensorflow as tf
from Measure import *


def GenEpisode_Imiation(score, label):
    ndoc = len(score)
    labeltmp = label.tolist()

    positions = range(ndoc)
    ranklist = np.zeros(ndoc, dtype=np.int32)
    for position in range(ndoc):
        choice = np.argmax(labeltmp)
        ranklist[position] = positions[choice]
        del labeltmp[choice]
        del positions[choice]
    return ranklist

def GenEpisode_Softmax(score):
    ndoc = len(score)
    scoretmp = score.tolist()

    positions = range(ndoc)
    ranklist = np.zeros(ndoc, dtype=np.int32)
    for position in range(ndoc):
        policy = np.exp(scoretmp) / np.sum(np.exp(scoretmp))
        choice = np.random.choice(len(policy), 1, p=policy)[0]
        ranklist[position] = positions[choice]

        del scoretmp[choice]
        del positions[choice]

    return ranklist



###################### basic class ########################################
class RL_BP(object):
    def __init__(self, Nhidden_unit, Nfeature, Learningrate, Lenepisode, Resultfile):

        self.Nfeature = Nfeature
        self.Lenepisode = Lenepisode

        self.W = np.random.rand(Nfeature)
        # self.W = np.zeros(Nfeature)
        self.lr = Learningrate

        self.resultfile = Resultfile

        self.Ntop = 10
        self.memory = []
        self.ite=0

        self.Nhidden_unit = Nhidden_unit

        global scores, input_docs, position, learning_rate, sess, train_step, cross_entropy, grads_vars, prob

        input_docs = tf.placeholder(tf.float32, [None, self.Nfeature])
        position = tf.placeholder(tf.int64)
        learning_rate = tf.placeholder(tf.float32, shape=[])

        # Generate hidden layer
        W1 = tf.Variable(tf.truncated_normal([self.Nfeature, self.Nhidden_unit], stddev=0.1 / np.sqrt(float(Nfeature))))
        # b1 = tf.Variable(tf.zeros([1, hidden_units]))
        h1 = tf.tanh(tf.matmul(input_docs, W1))

        # Second layer -- linear classifier for action logits
        W2 = tf.Variable(tf.truncated_normal([self.Nhidden_unit, 1], stddev=0.1 / np.sqrt(float(self.Nhidden_unit))))
        # b2 = tf.Variable(tf.zeros([1]))
        scores = tf.transpose(tf.matmul(h1, W2))  # + b2
        prob = tf.nn.softmax(scores)

        init = tf.global_variables_initializer()
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prob, labels=position)
        # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        grads_vars = opt.compute_gradients(cross_entropy)
        train_step = opt.apply_gradients(grads_vars)

        # Start TF session
        sess = tf.Session()
        sess.run(init)
        # train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    def GenEpisodes(self, Queryids, Data):
        thendcg = np.zeros(self.Ntop)
        thedcg = np.zeros(self.Ntop)
        themap = 0.0
        thealldcg = 0.0

        for queryid in Queryids:
            ### gen episode
            QueryInfo = Data[queryid]
            score = sess.run([scores], feed_dict={input_docs: QueryInfo['feature']})[0].reshape([-1])
            ranklist = GenEpisode_Imiation(score, QueryInfo['label'])
            reward = GetReturn_DCG(QueryInfo['label'][ranklist])
            self.memory.append({'queryid': queryid, 'score': score, 'reward': reward, 'ranklist': ranklist})

            ### measure
            rates = QueryInfo['label'][ranklist]
            themap += MAP(rates)
            thendcg += NDCG(self.Ntop, rates)
            thedcg += DCG(self.Ntop, rates)
            thealldcg += DCG_all(rates)

        nquery = len(Queryids)
        themap = themap / nquery
        thendcg = thendcg / nquery
        thedcg = thedcg / nquery
        thealldcg = thealldcg / nquery
        print  'jiayou: ', themap, thealldcg, thendcg[0], thendcg[2], thendcg[4], thendcg[9], thedcg[0], thedcg[2], thedcg[4], thedcg[9]

        info = yaml.dump({"ite": self.ite, "type": type, "MAP": themap, "Return": thealldcg, "DCG": thedcg, "NDCG": thendcg})+'\n'
        self.resultfile.write(info)

    def UpPolicy(self, Data):
        self.ite +=1

        for item in self.memory:
            queryid = item['queryid']
            reward = item['reward']
            ranklist = item['ranklist']
            QueryInfo = Data[queryid]

            ## top K
            ndoc = len(ranklist)
            lenghth = min(self.Lenepisode, ndoc)

            for pos in range(lenghth):
                loss, _ = sess.run([cross_entropy, train_step], feed_dict={input_docs: QueryInfo['feature'][ranklist], position: [0], learning_rate: self.lr * reward[pos]})
                ranklist = np.delete(ranklist, 0)

        del self.memory[:]

    def Eval(self, Data, type):
        thendcg = np.zeros(self.Ntop)
        thedcg = np.zeros(self.Ntop)
        themap = 0.0
        thealldcg = 0.0
        for queryid in Data.keys():
            QueryInfo = Data[queryid]
            score = sess.run(scores, feed_dict={input_docs: QueryInfo['feature']})[0].reshape([-1])
            rates = QueryInfo['label'][np.argsort(score)[np.arange(len(score) - 1, -1, -1)]]
            themap += MAP(rates)
            thendcg += NDCG(self.Ntop, rates)
            thedcg += DCG(self.Ntop, rates)
            thealldcg += DCG_all(rates)

        nquery = len(Data.keys())
        themap = themap / nquery
        thendcg = thendcg / nquery
        thedcg = thedcg /nquery
        thealldcg = thealldcg / nquery

        info=yaml.dump({"ite":self.ite, "type": type, "MAP": themap, "Return": thealldcg, "DCG": thedcg, "NDCG": thendcg})+'\n'
        self.resultfile.write(info)

        print type, ':  ', themap, thealldcg, thendcg[0], thendcg[2], thendcg[4], thendcg[9], thedcg[0], thedcg[2], thedcg[4], thedcg[9]




##########################  BP score function #################################
class RL_Imi_BP(RL_BP):

    def GenEpisodes(self, Queryids, Data):
        thendcg = np.zeros(self.Ntop)
        thedcg = np.zeros(self.Ntop)
        themap = 0.0
        thealldcg = 0.0

        for queryid in Queryids:
            ### gen episode
            QueryInfo = Data[queryid]
            score = sess.run([scores], feed_dict={input_docs: QueryInfo['feature']})[0].reshape([-1])
            ranklist = GenEpisode_Imiation(score, QueryInfo['label'])
            reward = GetReturn_DCG(QueryInfo['label'][ranklist])
            self.memory.append({'queryid': queryid, 'score': score, 'reward': reward, 'ranklist': ranklist})

            ### measure
            rates = QueryInfo['label'][ranklist]
            themap += MAP(rates)
            thendcg += NDCG(self.Ntop, rates)
            thedcg += DCG(self.Ntop, rates)
            thealldcg += DCG_all(rates)

        nquery = len(Queryids)
        themap = themap / nquery
        thendcg = thendcg / nquery
        thedcg = thedcg / nquery
        thealldcg = thealldcg / nquery
        print  'jiayou: ', themap, thealldcg, thendcg[0], thendcg[2], thendcg[4], thendcg[9], thedcg[0], thedcg[2], thedcg[4], thedcg[9]

        info = yaml.dump({"ite": self.ite, "type": type, "MAP": themap, "Return": thealldcg, "DCG": thedcg, "NDCG": thendcg})+'\n'
        self.resultfile.write(info)


class RL_Softmax_BP(RL_BP):

    def GenEpisodes(self, Queryids, Data):
        thendcg = np.zeros(self.Ntop)
        thedcg = np.zeros(self.Ntop)
        themap = 0.0
        thealldcg = 0.0

        for queryid in Queryids:
            ### gen episode
            QueryInfo = Data[queryid]
            score = sess.run([scores], feed_dict={input_docs: QueryInfo['feature']})[0].reshape([-1])
            ranklist = GenEpisode_Softmax(score)

            reward = GetReturn_DCG(QueryInfo['label'][ranklist])
            self.memory.append({'queryid': queryid, 'score': score, 'reward': reward, 'ranklist': ranklist})

            ### measure
            rates = QueryInfo['label'][ranklist]
            themap += MAP(rates)
            thendcg += NDCG(self.Ntop, rates)
            thedcg += DCG(self.Ntop, rates)
            thealldcg += DCG_all(rates)

        nquery = len(Queryids)
        themap = themap / nquery
        thendcg = thendcg / nquery
        thedcg = thedcg / nquery
        thealldcg = thealldcg / nquery
        print  'jiayou: ', themap, thealldcg, thendcg[0], thendcg[2], thendcg[4], thendcg[9], thedcg[0], thedcg[2], thedcg[4], thedcg[9]


        info = yaml.dump({"ite": self.ite, "type": type, "MAP": themap, "Return": thealldcg, "DCG": thedcg, "NDCG": thendcg})+'\n'
        self.resultfile.write(info)



