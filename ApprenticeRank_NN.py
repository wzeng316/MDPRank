
import tensorflow as tf
import time

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

        self.W = np.random.rand(Nfeature)
        # self.W = np.zeros(Nfeature)
        self.lr = Learningrate

        self.resultfile = open(Resultfile, 'w')
        self.resultfile_w = open(Resultfile + '_W', 'w')

        self.Ntop = 10
        self.memory = []

        hidden_units = 10

        global scores, input_docs, position, learning_rate, sess, train_step, cross_entropy, grads_vars, prob

        input_docs = tf.placeholder(tf.float32, [None, self.Nfeature])
        position = tf.placeholder(tf.int64)
        learning_rate = tf.placeholder(tf.float32, shape=[])

        # Generate hidden layer
        W1 = tf.Variable(tf.truncated_normal([self.Nfeature, hidden_units], stddev=0.1 / np.sqrt(float(Nfeature))))
        # b1 = tf.Variable(tf.zeros([1, hidden_units]))
        h1 = tf.tanh(tf.matmul(input_docs, W1))

        # Second layer -- linear classifier for action logits
        W2 = tf.Variable(tf.truncated_normal([hidden_units, 1], stddev=0.1 / np.sqrt(float(hidden_units))))
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
        themap = 0.0

        nquery = len(Queryids)

        for queryid in Queryids:
            QueryInfo = Data[queryid]

            score = sess.run([scores], feed_dict={input_docs: QueryInfo['feature']})[0].reshape([-1])

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

        themap = themap / nquery
        thendcg = thendcg / nquery

        print  'train: ', themap, thendcg[0], thendcg[2], thendcg[5], thendcg[9],

        self.resultfile.write(MeasureResult(themap, thendcg))

    def UpPolicy(self, Data):

        for item in self.memory:
            queryid = item['queryid']
            score = item['score']
            reward = item['reward']
            ranklist = item['ranklist']

            QueryInfo = Data[queryid]

            ndoc = len(ranklist)

            lenghth = min(self.Lenepisode, ndoc)

            for pos in range(lenghth):
                loss, _ = sess.run([cross_entropy, train_step], feed_dict={input_docs: QueryInfo['feature'][ranklist], position: [0], learning_rate: self.lr * reward[pos]})

                # gradients_and_vars, prosb = sess.run([grads_vars, prob], feed_dict={input_docs:QueryInfo['feature'][ranklist], position:[0]})
                # for g, v in gradients_and_vars:
                #     if g is not None:
                #         print "****************this is variable*************"
                #         print v
                #         print "****************this is gradient*************"
                #         print g

                ranklist = np.delete(ranklist, 0)

        del self.memory[:]

    def Eval(self, Data, type):

        thendcg = np.zeros(self.Ntop)
        themap = 0.0

        nquery = len(Data.keys())

        for queryid in Data.keys():
            QueryInfo = Data[queryid]

            score = sess.run(scores, feed_dict={input_docs: QueryInfo['feature']})[0].reshape([-1])


            rates = QueryInfo['label'][np.argsort(score)[np.arange(len(score) - 1, -1, -1)]]

            themap += MAP(rates)
            thendcg += NDCG(self.Ntop, rates)

            # print NDCG/nquery
        themap = themap / nquery
        thendcg = thendcg / nquery
        self.resultfile.write(MeasureResult(themap, thendcg))

        if type == 'test':
            self.resultfile.write('\n')

        print type, ':  ', themap, thendcg[0], thendcg[2], thendcg[5], thendcg[9]


if __name__ == '__main__':

    dataset = 'MSLR-WEB10K'
    fold = sys.argv[1]
    print fold

    datafile = '/mnt/disk1/zengwei/Data/MSLR-WEB10K' + dataset + '/' + fold + '/'

    train_data = LoadData(datafile+'train.txt', dataset)
    vali_data  = LoadData(datafile+'vali.txt',  dataset)
    test_data  = LoadData(datafile+'test.txt',  dataset)



    nquery = len(train_data.keys())

    Nfeature=136
    Learningrate=0.001
    Nepisode=100

    Lenepisode=10

    Resultfile = 'ApprenticeRank/V1_2_'+dataset+'_'+fold+'_'+time.strftime("%m%d", time.localtime())


    learner = RL(Nfeature, Learningrate, Lenepisode, Resultfile)
    learner.Eval(train_data, 'train')
    learner.Eval(vali_data , 'vali')
    learner.Eval(test_data , 'test')
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