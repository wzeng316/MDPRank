import time
import yaml

from MDPRank import *
dataset = 'MSLR-WEB10K'

Ip_info = str(yaml.load(file(os.environ['HOME']+'/.host_info.yml'))['host'])
print Ip_info

data  = LoadData('data',  dataset)
nquery = len(data.keys())

Nfeature = 136
Learningrate = 0.0001
Nepisode = 100
Lenepisode = 1000000
Resultfile = open('ApprenticeRank/' + Ip_info + 'data' + dataset +  '_' + time.strftime("%m%d",time.localtime()),'w')
learner = RL_Imi_BP(Nfeature, Learningrate, Lenepisode, Resultfile)
learner.Eval(data, 'test')
# np.random.seed(datetime.datetime.now().microsecond)


for ite in range(100):
    batch = np.random.randint(nquery, size=Nepisode)

    Queryids = []
    for i in batch:
        Queryids.append(data.keys()[i])

    learner.GenEpisodes(Queryids, data)
    learner.UpPolicy(data)
    learner.Eval(data, 'train')


