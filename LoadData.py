import re
import numpy as np
import sys

def LoadData(Filename, dataset):
    # type: (object, object) -> object
    #open the file
    """

    :rtype: object
    """
    print Filename
    FILE=open(Filename,'r')
    if not FILE:
        print 'Open File failed \n'
        FILE.close()
        sys.exit(-2)
    
    # judge the dataset
    matchMQ  = re.match(r'^([Mm][Qq]+)',dataset)
    matchMS  = re.match(r'^([Mm][Ss]+)',dataset)
    print dataset, matchMS, matchMQ
    
    # load the data
    original_data={}
    docid=-1
    
    mean=np.zeros(136)
    
    for line in FILE:
        line=line.strip('\n')        
        if matchMQ:# MQ2007 MQ2008
            m = re.match(r'^(\d+) qid\:([^\s]+) (.*?) \#docid = ([^\s]+) inc = ([^\s]+) prob = ([^\s]+).$',line)
        elif matchMS:# MS
            m = re.match(r'^(\d+) qid\:([^\s]+) (.*?).$',line)
        else:# OHSUMED
            m = re.match(r'^(\d+) qid\:([^\s]+) (.*?) \#docid = ([^\s]+).$',line)
        
        # read the data
        if m:
            docid += 1
            
            label = m.group(1)
            label=int(label)
            
            queryid = m.group(2)
            
            feature=[]
            feature_str=m.group(3).strip().split(' ')
            for f in feature_str:
                feature.append(float(f.split(':')[1]))
            feature=np.asarray(feature)
            
            mean += feature
            

            if not original_data.has_key(queryid):
                original_data[queryid]={}
                
            if not original_data[queryid].has_key(label):
                original_data[queryid][label]={}
            
            if not original_data[queryid][label].has_key(docid):
                original_data[queryid][label][docid]= feature

        else:
            print 'Error to parse Feature at line \n'
            sys.exit(-2)
            
    mean=mean/docid
    
    sigma=np.zeros(136)
    for queryid in  original_data.keys():
        for label in  original_data[queryid].keys():
            for doc in original_data[queryid][label].keys():
                feature = original_data[queryid][label][doc]
                sigma += (feature-mean)*(feature-mean)
    sigma = sigma/docid
    sigma = np.sqrt(sigma)
    for i in range(len(sigma)):
        if sigma[i]==0:
            sigma[i]=1


    Data={}
    for queryid in  original_data.keys():
        label_matrix=[]
        feature_matrix=[]
        for label in  original_data[queryid].keys():
            for doc in original_data[queryid][label].keys():
                label_matrix.append(label)
                feature_matrix.append((original_data[queryid][label][doc]-mean)/sigma)

        label_matrix=np.asarray(label_matrix)
        feature_matrix=np.asarray(feature_matrix)

        Data[queryid]={'label':label_matrix, 'feature':feature_matrix}
                
    FILE.close()
    
    # print original_data
    
    return Data

    