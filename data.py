# -*- coding: utf-8 -*-
import os # 测试访问路径可能用到
import math
import random
import collections
from keras.datasets import mnist
import numpy as np
from sklearn import cluster
from collections import defaultdict
from operator import itemgetter
from itertools import permutations


# obtain sample data from keras datasets
def get_data():
    ''' get 70000 unlabeled data in sahpe (70000,784)'''

    (train_dt,train_lab),(test_dt,test_lab) = mnist.load_data()
    # in shape: (70000,28,28)
    unlabel_dt = np.concatenate((train_dt,test_dt),axis=0)
    lab = np.concatenate((train_lab,test_lab),axis=0)
    # for spectral clustering (SC) needed.
    unlabel_dt = unlabel_dt.reshape(len(unlabel_dt),784)
    return unlabel_dt,lab
    #。 ###
def SC(W,params):
    ''' Implementing spectral clustering algorithm on unlabel data in shape(sample_nums,784) or affinity matrix W'''
    spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'],
        eigen_solver='arpack',affinity=params['affinity'],eigen_tol=params['n_clusters'],n_neighbors=params['n_nbrs'])
    s = spectral.fit(W)
    label_pred = s.fit_predict(W)
    return label_pred

def get_pairs(dt,W,params):

    ''' get pairs according to label_pred by SC
        dt: data. data prepare to get pairs.
        W: None or affinity W,if None,let W = dt,defult value is None.
        return: pairs and pairs' label used to siamese network
    '''
    if W is not None:
        labels = SC(W,params)
        pairs,pairs_lab,class_indices, newcls_number = creat_pairs(dt,labels)
    else:
        W = dt
        labels = SC(W,params)
        pairs,pairs_lab,class_indices,newcls_number = creat_pairs(dt,labels)
    # shuffle pairs and corresponding labels
    index = np.random.permutation(pairs.shape[0])
    pairs = pairs[index]
    pairs_lab = pairs_lab[index]
    return pairs,pairs_lab,class_indices,newcls_number

def creat_pairs(data,labels):
    categories = len(collections.Counter(labels))
    print('categories is :', categories)
    class_indices = [np.where(labels == i)[0] for i in range(categories)]
    np.save('class_indices.npy',class_indices)
    pairs = [] #一会儿一对对的样本要放在这里
    pairs_lab = [] # 这是对的样本，但是不是数据的样本

    # 下面的两步其实都是为了起“过滤”作用，希望得到每一类下更纯的数据。
    new_class_indices = get_new_idx(data,class_indices)
    # new_class_indices = cluster_filter(data,new_class_indices)
    np.save('new_idx.npy',new_class_indices)
    # 类别数目：
    ################################
    newcls_number = np.array(new_class_indices).shape[0] # number of category
    # get pos pairs:
    pos_pairs_temp = []
    for i in range(newcls_number):
        pairs_generator = permutations(new_class_indices[i],2)
        pairs = [[data[items1],data[items2]] for (items1,items2) in pairs_generator]
        pos_pairs_temp.append(pairs)
    pos_pairs = []
    for i in range(newcls_number):
        for p_ in pos_pairs_temp[i]:
            pos_pairs.append(p_)

    # get neg pairs
    neg_pairs = []
    # 传统for循环的迭代是有1000次限制的，要用itertools 中的迭代工具包才行
    neg_num = np.arange((len(pos_pairs))) # 这里的重复次数可能要上万次
    for _ in neg_num:
        c_1 = random.choice(np.arange(newcls_number))
        inc = random.randrange(1,newcls_number)
        c_2 = (c_1+inc) % newcls_number
        index1 = random.choice(new_class_indices[c_1])
        index2 = random.choice(new_class_indices[c_2])
        neg_pairs.append([data[index1],data[index2]])

    pos_pairs = np.array(pos_pairs)
    neg_pairs = np.array(neg_pairs)
    pos_lab = np.ones(len(pos_pairs))
    neg_lab = np.zeros(len(neg_pairs))
    pairs = np.concatenate((pos_pairs,neg_pairs),axis=0)
    pairs_lab = np.concatenate((pos_lab,neg_lab))
    
    shuffle = np.random.permutation(pairs.shape[0])
    pairs = pairs[shuffle]
    pairs_lab = pairs_lab[shuffle]
    if pairs.shape[0] < 40000:
        pass
    else:
        pairs = pairs[:40000]
        pairs_lab = pairs_lab[:40000]
    print('pairs shape is:',pairs.shape)
    print('pairs label shape is:' , pairs_lab.shape)
    return pairs, pairs_lab,class_indices, newcls_number

def get_new_idx(data, class_indices):
    '''compute the means and centroid of the position of each category in the class_indices.
    data is all_data used to clustering
    a nested indices 2D array of data,e.g. the data's indices we think belonging to the same class.
    '''
    # cent= [] # 存放图心
    
    n_idx_dict = defaultdict(set)
    for i in range(np.array(class_indices).shape[0]):
        # n = len(class_indices[i])
        # 获取第i类的形心、
        cent=np.mean(data[class_indices[i]], axis= 0)
        dist_idx=[] # 存放索引-距离
        for idx in class_indices[i]:
            
            dist = distance(data[idx],cent)
            dist_idx.append((idx,dist))
        dist_idx=sorted(dist_idx,key=itemgetter(1)) # 此时这是一个按距离排序的[(索引，距离),...]数据结构
        dist_idx=dist_idx[:int(len(dist_idx)/2)] # 取前面最近的一半数据的（索引，dist）
        for elements in dist_idx:
            n_idx_dict[i].add(elements[0])
    new_idx_array = np.array(transDicToList(n_idx_dict))
    new_idx_array = cluster_filter(data,new_idx_array)
    return new_idx_array

def cluster_filter(data,index):
    ''' only closest points can be selected'''
    dist_idx = []
    for i in range(index.shape[0]):
        cent=np.mean(data[index[i]],axis=0)
        dist2_list = []
        for idx in index[i]:
            dist=distance(data[idx],cent)
            dist2_list.append(dist)
        d_i = np.mean(dist2_list)
        dist_idx.append((i,d_i))
    dist_idx = sorted(dist_idx,key=itemgetter(1))
    row_num = int(math.ceil(3*index.shape[0]/4.0))
    dist_idx = dist_idx[:row_num]
    new_idx = []
    for elements in dist_idx:
        new_idx.append(elements[0])
    index = index[new_idx]
    return index

def transDicToList(x):
    '''
    transform the multiple value dict into list, in order to find out the proble pair data index
    x: a multivalue dict,e.g. a key with multiple value
    '''
    new_list=[]
    for i in list(x.values()):
        new_list.append(list(i))
    return new_list

def distance(x1,x2):
    '''
    compute distance of the to array object x1 and x2.
    only used to purify the cluster'''
    return np.sqrt(np.sum(np.square(x1-x2)))

def batch_generator(pairs,pairs_label,batch_size):
    # 这只是一个生成器函数，调用这个函数只是返回一个生成器对象，而不是数据哦！！
    # 可以在for循环中调用生成器对象。
    assert pairs.shape[0] == pairs_label.shape[0],('pairs.shape:%s labels.shape:%s' % (pairs.shape,labels.shape))
    num_examples = len(pairs) # 样本数量
    batch_num = int(len(pairs)/batch_size) + 1 # 在训练集上遍历一遍（一轮）需要的batch数量
    for num in range(batch_num):
        start_index = num * batch_size
        end_index = min((num+1)*batch_size,num_examples)
        # random_indx = np.random.permutation(len(pairs))  # 具有打乱数据功能的
        # permut = random_indx[start_index:end_index]
        # 如果数据输入前已经打乱了，就没必要再去打乱了
        # 多维数组切片，先取出第一维的（即哪些pairs、pairs_labels)，再选择其他维的(比如每一维的第一维或者第二维等等)。
        batch_slice = np.arange(start_index,end_index) 
        x1, x2 = pairs[batch_slice,0],pairs[batch_slice,1]
        y = pairs_label[batch_slice]
        yield ([x1,x2],y)
