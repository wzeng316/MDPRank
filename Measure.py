import numpy as np

hsNdcgRelScore = {'4':15,'3':7,'2':3,'1':1,'0':0}
hsPrecisionRel={'4':1,'3':1,'2':1,'1':1,'0':0}
iMaxPosition = 10
topN = 10

def MeasureW_onequery(w, Querys, original):
    
    nfeature=w.size
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
        label_matrix = np.asarray(labels)
        # print labels
        
        score = np.dot(feature_matrix, w)
        rates=sort(score, label_matrix)
        # print rates
        
        
        #print rates
        themap = MAP(rates)
        meanmap = meanmap + themap        
        theNDCG = NDCG(topN, rates)
        meanNDCG = meanNDCG + theNDCG
    
    return meanmap/n_query, meanNDCG/n_query


def MeasureW(w, query_in_original):
    
    labels=[]
    features=[]
    
    for label in original[queryid].keys():
        for doc in original[queryid][label].keys():
            labels.append(label)
            features.append(original[queryid][label][doc])
    
    feature_matrix = np.asarray(features)
    label_matrix = np.asarray(labels)

    score = np.dot(feature_matrix, w)
    rates=sort(score, label_matrix)
        
    #print rates
    themap = MAP(rates)
    theNDCG = NDCG(topN, rates)
    
    return themap, themap


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
    #print avgPrecision, numRelevant, avgPrecision / numRelevant
    return avgPrecision / numRelevant

def NDCG(topN, rates):
    ndcg = np.zeros(topN)
    dcg = DCG(topN,rates)


    stRates = sorted(rates,key=f,reverse=True)
    bestDcg = DCG(topN,stRates)
    #print dcg[0], bestDcg[0]
            
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


def GetReturn(rates):
    gain = np.zeros(len(rates))

    thegain = 1.0 * hsNdcgRelScore[str(rates[0])]
    for iPos in range(1, len(rates)):
        r = 0
        if (iPos < len(rates)):
            r = rates[iPos]
        else:
            r = 0

        if (iPos < 2):
            thegain += hsNdcgRelScore[str(r)]
        else:
            thegain += round(hsNdcgRelScore[str(r)] * np.log(2.0) / np.log(iPos + 1.0), 6)

    for iPos in range(len(rates)):
        gain[iPos] = thegain

    # if np.sum(gain * gain)>0:
    #     gain = gain / np.sqrt(np.sum(gain*gain))

    return gain