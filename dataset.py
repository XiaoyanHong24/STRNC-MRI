import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
import glob
path_0=glob.glob("dataset/T0/0/*.png")
display(path_0[0:3],len(path_0))
path_1=glob.glob("dataset/T0/1/*.png")
display(path_1[0:3],len(path_1))
path_2=glob.glob("dataset/T0/2/*.png")
display(path_2[0:3],len(path_2))
path_3=glob.glob("dataset/T0/3/*.png")
display(path_3[0:3],len(path_3))
train_num=4

#number of examples
num_examples=90

#image width
img_width=56

#image height
img_height=56
channels=1
train_dataset=np.zeros([train_num,num_examples,img_height,img_width,channels],dtype=np.float32)
train_dataset.shape
a=0
b=1
c=2
d=3
ran0=np.random.permutation(len(path_0))[:num_examples]
for index,name in enumerate(ran0):
    img_1=tf.io.read_file(path_0[name])
    img_1=tf.image.decode_png(img_1,channels=channels)
    img_1=tf.image.resize(img_1,[img_width,img_height])
    img_1=img_1/255
    train_dataset[a,index]=img_1

ran1=np.random.permutation(len(path_1))[:num_examples]
for index,name in enumerate(ran1):
    img_1=tf.io.read_file(path_1[name])
    img_1=tf.image.decode_png(img_1,channels=channels)
    img_1=tf.image.resize(img_1,[img_width,img_height])
    img_1=img_1/255
    train_dataset[b,index]=img_1

ran2=np.random.permutation(len(path_2))[:num_examples]
for index,name in enumerate(ran2):
    img_1=tf.io.read_file(path_2[name])
    img_1=tf.image.decode_png(img_1,channels=channels)
    img_1=tf.image.resize(img_1,[img_width,img_height])
    img_1=img_1/255
    train_dataset[c,index]=img_1

ran3=np.random.permutation(len(path_3))[:num_examples]
for index,name in enumerate(ran3):
    img_1=tf.io.read_file(path_3[name])
    img_1=tf.image.decode_png(img_1,channels=channels)
    img_1=tf.image.resize(img_1,[img_width,img_height])
    img_1=img_1/255
    train_dataset[d,index]=img_1
train_dataset.shape


def few_shot(train_dataset):
    support_set=np.zeros([batch,num_way*num_shot,img_height,img_width,channels],
                         dtype=np.float32)
    query_set=np.zeros([batch,num_way*num_query,img_height,img_width,channels],
                       dtype=np.float32)
    support_labels=np.zeros([batch,num_way*num_shot],dtype=np.int32)
    query_labels=np.zeros([batch,num_way*num_query],dtype=np.int32)
    for i in range(batch):
        episodic_classes=np.random.permutation(train_num)[:num_way]
        support=np.zeros([num_way,num_shot,img_height,img_width,channels],dtype=np.float32)
        query=np.zeros([num_way,num_query,img_height,img_width,channels],dtype=np.float32)

        for index,class_ in enumerate(episodic_classes):

            selected=np.random.permutation(num_examples)[:num_shot+num_query]

            support[index]=train_dataset[class_,selected[:num_shot]]

            query[index]=train_dataset[class_,selected[num_shot:]]

        support=np.expand_dims(support,axis=-1).reshape(
            num_way*num_shot,img_width,img_height,channels)
        query=np.expand_dims(query,axis=-1).reshape(
            num_way*num_query,img_width,img_height,channels)
        support_set[i]=support
        query_set[i]=query

        s_labels=[]
        q_labels=[]
        for j in range(num_way):
            s_labels=s_labels+[episodic_classes[j]]*num_shot
            q_labels=q_labels+[episodic_classes[j]]*num_query
        support_labels[i]=np.array(s_labels)
        query_labels[i]=np.array(q_labels)
    return support_set,query_set,support_labels,query_labels