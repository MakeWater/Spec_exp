# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time
import tensorflow as tf
import numpy as np 
from keras.engine.training import _make_batches
from sklearn import cluster
import network
from data import get_data, get_pairs, batch_generator

params = {'n_clusters':23, 'n_nbrs':27, 'affinity':'nearest_neighbors'}
if not os.path.exists('log'):
    os.mkdir('log')
log_dir = 'log'
# unlabel_data, _ = get_data()
unlabel_data = np.load('unlabdt.npy')
unlabel_data = unlabel_data[:1000] ############## for test !!!!!!!!!!!

sess = tf.InteractiveSession()

# siam.initialize
# siam.merged
# saver = siam.saver



siam = network.siamese()

# global_step = tf.Variable(0,trainable=False)
# with tf.name_scope('learning_rate'):
#     learning_rate = tf.train.exponential_decay(0.01,global_step,50,0.96)
#     tf.summary.scalar('learning_rate',learning_rate)

# with tf.name_scope('loss'):
#     loss = siam.loss
#     tf.summary.scalar('loss',loss)

# train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# merged = tf.summary.merge_all()

# tf.global_variables_initializer().run()

# simi,simi1,simi2 = sess.run([siam.similarity,siam.output1,siam.output2],feed_dict={siam.x1:np.expand_dims(unlabel_data[358],axis=0),
#                                                        siam.x2:np.expand_dims(unlabel_data[418],axis=0)})
# print(simi,simi1,simi2)
# tf.global_variables_initializer().run()


batch_size = 1


for game_epoch in range(2):
    if game_epoch == 0:
        epoches = 20 # 训练轮次
        global_step = tf.Variable(0,trainable=False)
        
        with tf.name_scope('loss'):
            loss = siam.loss
            tf.summary.scalar('loss',loss)

        with tf.name_scope('train'):
            with tf.name_scope('learning_rate'):
                learning_rate = tf.train.exponential_decay(0.001,global_step,1000,0.96,staircase=True)
                tf.summary.scalar('learning_rate',learning_rate)
            with tf.name_scope('train_step'):
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step)



        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir + '/train',sess.graph)

        W = None
        # pairs, pairs_label,newcls_number = get_pairs(unlabel_data,W,params)
        # params['n_clusters'] = newcls_number
        pairs = np.load('pairs.npy')
        pairs_label = np.load('pairs_label.npy')

        print('pairs shape is:',pairs.shape)
        print('pairs label shape is:',pairs_label.shape)
        time.sleep(1)
        # np.save('pairs.npy',pairs)
        # np.save('pairs_label.npy',pairs_label)
        # steps_each_epoch = int(math.ceil(float(len(pairs)) / batch_size)) # 每一轮的步数
        for epoch in range(epoches):
            print('The next epoch is ....',epoch)
            time.sleep(1)
            # shuffle the pairs each epoch
            shuffle = np.random.permutation(pairs.shape[0])
            pairs = pairs[shuffle]
            pairs_label = pairs_label[shuffle]
            data_generator = batch_generator(pairs,pairs_label,batch_size)
            steps = 0
            # 这个for循环只是用来读取数据的。
            for ([batch_x1,batch_x2],y_true) in data_generator:# get batch data from data generator
                x1 = batch_x1
                x2 = batch_x2
                y_true = y_true

                summary, _, losses, s = sess.run([merged, train_step,siam.loss,siam.similarity], feed_dict={
                                    siam.x1: x1,
                                    siam.x2: x2,
                                    siam.y_true: y_true,
                                    siam.dropout:0.4})
                if steps%100==0:
                    train_writer.add_summary(summary,steps)
                    # saver.save(sess,os.path.join(log_dir + 'model','model.ckpt'),steps)# save trained model.
                    print('the game_epoch is %d,epoch is %d,step is %d,loss is %.3f, the steps similarity is %.8f' % 
                        (game_epoch, epoch, steps,losses,s))
                steps += 1
        train_writer.close()



    elif game_epoch == 1:
        # 1、准备数据：W，pairs，pairs label, 修改params
        # compute the affinity matrix by siamese:
        n = len(unlabel_data)
        W = np.zeros((n,n))
        start = time.clock()
        for i in range(n):
            for j in range(i,n):
                if i == j:
                    W[i][j] = 1
                else:
                    # approch 2:(solve the shape problem by array = np.expand_dims(array,axis=0)
                    w = sess.run(siam.similarity,feed_dict={siam.x1:np.expand_dims(unlabel_data[i],axis=0),
                        siam.x2:np.expand_dims(unlabel_data[j],axis=0),siam.dropout:1.0})
                    # 矩阵稀疏化：
                    W[i][j] = 1 if w >= 0.65 else 0
                    if j%100 == 0:
                        print('the similarity between {} and {} is {}'.format(i,j,W[i][j]))
        elapsed = (time.clock() - start)
        print('Time used to compute affinity :',elapsed)
        # 转置成对称阵
        W = W + W.transpose()
        np.save('W_{}.npy'.format(game_epoch),W) # W 转为对称阵 
        params['affinity'] = 'precomputed'
        print('AFFINITY HAS BEEN COMPUTED AND SAVED ! ##########################################################')
        

        # 预测新的对
        pairs1,pairs_label_1,class_indices,newcls_number = get_pairs(unlabel_data,W,params)
        # 存储索引：
        np.save('c_idx{}.npy',class_indices)
        np.save('n_idx{}.npy',newcls_number)
        # 迭代预测的类别，递减（到时候也可以通过查看newcls_number的形状来看递减过程）：
        if newcls_number < 10:
            newcls_number = 10
        params['n_clusters'] = newcls_number
        # params['n_nbrs'] = None
        print('pairs1 shape is:',pairs1.shape)
        print('pairs1 label shape is:',pairs_label_1.shape)
        # save pairs data:
        np.save('pairs{}.npy'.format(game_epoch),pairs1)
        np.save('pairs_label_{}.npy'.format(game_epoch),pairs_label_1)
        time.sleep(5)
        
        #本质上是用上一次网络计算的矩阵预测的对来进行下一次的网络训练：

        with tf.name_scope('learning_rate'):
            learning_rate = tf.train.exponential_decay(0.01,global_step,500,0.96,staircase=True) # 阶梯式更新学习率
            tf.summary.scalar('learning_rate{}'.format(game_epoch),learning_rate)

        with tf.name_scope('loss'):
            loss = siam.loss
            tf.summary.scalar('loss{}'.format(game_epoch),loss)

        with tf.name_scope('train_step'):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step)        
        # train process use affinity computed by siamese network.

        tf.global_variables_initializer().run()
        merged = tf.summary.merge_all()

        test_writer = tf.summary.FileWriter(log_dir + '/test{}'.format(game_epoch))
        print('game_epoch',game_epoch)
        epoches = 10
        for epoch2 in range(epoches):
            shuffle = np.random.permutation(pairs1.shape[0])
            pairs1 = pairs1[shuffle]
            pairs_label_1 = pairs_label_1[shuffle]
            data_generator2 = batch_generator(pairs1,pairs_label_1,batch_size)
            step_2 = 0
            # 这个for循环只是用来读取数据的。
            for ([batch_x1,batch_x2],y_true) in data_generator2:
                x1 = batch_x1
                x2 = batch_x2
                y_true = y_true
                
                summary, _, losses, s = sess.run([merged, train_step,loss,siam.similarity], feed_dict={
                                    siam.x1: x1,
                                    siam.x2: x2,
                                    siam.y_true: y_true,
                                    siam.dropout:0.4})
                if step_2 % 100 == 0:
                    test_writer.add_summary(summary,step_2)
                    # saver.save(sess,os.path.join(log_dir + 'model','model.ckpt'),steps)# save trained model.
                    print('the game_epoch is %d,epoch is %d,step is %d,loss is %.3f, the steps similarity is %.8f' % (game_epoch, epoch, step_2,losses,s))
                step_2 += 1
        test_writer.close()
print('Done Training!')