import numpy as np

hsNdcgRelScore = {'4':15,'3':7,'2':3,'1':1,'0':0}
hsPrecisionRel={'4':1,'3':1,'2':1,'1':1,'0':0}
iMaxPosition = 10
topN = 10

def MeasureW_onequery(w, Querys, original):
    n_query=len(Querys)
    meanmap = 0
    meanNDCG = np.zeros(topN)

    for queryid in Querys:
        labels=[]
        features=[]

        for label in original[queryid].keys():
            for doc in original[queryid][label].keys():
                labels.append(label)
                features.append(original[queryid][label][doc])

        feature_matrix = np.asarray(features)
        label_matrix = np.asarray(labels)  # print labels

        score = np.dot(feature_matrix, w)
        rates=sort(score, label_matrix)

        themap = MAP(rates)
        meanmap = meanmap + themap
        theNDCG = NDCG(topN, rates)
        meanNDCG = meanNDCG + theNDCG
    return meanmap/n_query, meanNDCG/n_query


def MeasureW(W, original_data):
    thendcg = np.zeros(topN)
    themap = 0.0
    nquery = len(original_data.keys())
    for queryid in original_data.keys():
        QueryInfo = original_data[queryid]
        score = np.dot(QueryInfo['feature'], W)
        rates = QueryInfo['label'][np.argsort(score)[np.arange(len(score) - 1, -1, -1)]]
        themap += MAP(rates)
        thendcg += NDCG(topN, rates)
    themap = themap / nquery
    thendcg = thendcg / nquery
    return themap, thendcg


def MAP(rates):
    numRelevant = 0.0
    avgPrecision = 0.0
    for iPos in range(len(rates)):
        if  hsPrecisionRel[str(rates[iPos])] == 1:
            numRelevant+=1
            #avgPrecision += round((numRelevant / (iPos + 1)),6 )
            avgPrecision += (numRelevant / (iPos + 1))
    if  numRelevant == 0:
        return 0.0
    return avgPrecision / numRelevant

def NDCG(topN, rates):
    ndcg = np.zeros(topN)
    dcg = DCG(topN,rates)
    stRates = sorted(rates,key=f,reverse=True)
    bestDcg = DCG(topN,stRates)
    iPos =0
    while iPos < topN and iPos < len(rates):
        if (bestDcg[iPos] != 0):
            ndcg[iPos] = float('%.6f'%(dcg[iPos] / bestDcg[iPos] ))
        iPos+=1
    return ndcg

def f(x):
    return hsNdcgRelScore[str(x)]

def DCG(topN,rates):
    dcg=[0.0]*topN
    dcg[0] = 1.0*hsNdcgRelScore[str(rates[0])]
    for iPos in range(1,topN):
        r=0
        if (iPos < len(rates)):
            r = rates[iPos]
        else:
            r = 0
        if (iPos < 2):
            dcg[iPos] = dcg[iPos - 1] + hsNdcgRelScore[str(r)]
        else:
            dcg[iPos] = dcg[iPos - 1] + round(hsNdcgRelScore[str(r)] * np.log(2.0) / np.log(iPos + 1.0),6)
    return dcg

def sort(score, label):
    n=len(score)
    for i in range(0,n):
        # print i
        for j in range(i,n):
            if score[i]<score[j]:
                tmp=score[j];
                score[j]=score[i]
                score[i]=tmp;

                tmp=label[j]
                label[j]=label[i]
                label[i]=tmp
    return label



###################################   DCG #########################################################################
def GetReward_DCG(rates):
    reward = np.zeros(len(rates))
    for iPos in range(len(rates)):
        if iPos < len(rates):
            r = rates[iPos]
        else:
            r = 0
        if iPos < 2:
            reward[iPos] = hsNdcgRelScore[str(r)]
        else:
            reward[iPos] = round(hsNdcgRelScore[str(r)] * np.log(2.0) / np.log(iPos + 1.0), 6)
    return  reward

def GetReturn_DCG(rates):
    ndoc = len(rates)
    returns = GetReward_DCG(rates)
    for iPos in range(len(rates)-1):
        returns[ndoc -2 - iPos] += returns[ndoc -1 - iPos]
    return returns

###################################   0-1  ##################################################################
def GetReward_Step(rates):
    reward = np.zeros(len(rates))
    for iPos in range(len(rates)):
        if iPos < len(rates):
            r = rates[iPos]
        else:
            r = 0
        if r ==0:
            reward[iPos] = 0
        else:
            reward[iPos] = 1
    return  reward

def GetReturn_Step(rates):
    ndoc = len(rates)
    returns = GetReward_Step(rates)
    for iPos in range(len(rates)-1):
        returns[ndoc -2 - iPos] += returns[ndoc -1 - iPos]
    return returns


###################################   Precision  ##################################################################
def GetReward_Precision(rates):
    reward = np.zeros(len(rates))
    for iPos in range(len(rates)):
        if iPos < len(rates):
            r = rates[iPos]
        else:
            r = 0
        if r ==0:
            reward[iPos] = 0
        else:
            reward[iPos] = 1
    return  reward

def GetReturn_Precision(rates):
    ndoc = len(rates)
    returns = GetReward_Precision(rates)
    for iPos in range(len(rates)-1):
        returns[ndoc -2 - iPos] += returns[ndoc -1 - iPos]
    return returns