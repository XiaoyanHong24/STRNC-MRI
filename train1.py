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
#number of examples
num_examples=90

#number of classes
num_way=4

num_shot=1

num_query=1

batch=32
def few_shot(train_dataset):
    support_set=np.zeros([batch,num_way*num_shot,img_height,img_width,channels],
                         dtype=np.float32)
    query_set=np.zeros([batch,num_way*num_query,img_height,img_width,channels],
                       dtype=np.float32)
    support_labels=np.zeros([batch,num_way*num_shot],dtype=np.int32)
    query_labels=np.zeros([batch,num_way*num_query],dtype=np.int32)
    for i in range(batch):
        episodic_classes=np.random.permutation(train_num)[:num_way]
        #支持集
        support=np.zeros([num_way,num_shot,img_height,img_width,channels],dtype=np.float32)
        #查询集
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
support_set,query_set,support_labels,query_labels=few_shot(train_dataset)
display(support_set.shape,query_set.shape,support_labels.shape,query_labels.shape)
model_cnn=tf.keras.Sequential()
model_cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=7,padding='same',
                                     activation='relu'))
model_cnn.add(tf.keras.layers.BatchNormalization())
model_cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model_cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=5,padding='same',
                                 activation='relu'))
model_cnn.add(tf.keras.layers.BatchNormalization())
model_cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model_cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',
                                 activation='relu'))
model_cnn.add(tf.keras.layers.BatchNormalization())
model_cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model_cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=1,padding='same',
                                 activation='relu'))
model_cnn.add(tf.keras.layers.BatchNormalization())
model_cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model_cnn.add(tf.keras.layers.Flatten())
model_cnn.add(tf.keras.layers.Dense())
model_lstm=tf.keras.Sequential()
model_lstm.add(tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(64,return_sequences=True)))
def cosine_similarity(support,query):
    eps=1e-10
    similarities=tf.zeros([support_samples,batch],tf.float32)
    i_sample=0
    for support_image in support:
        sum_support=tf.reduce_sum(tf.square(support_image),axis=1)
        support_magnitude=tf.math.rsqrt(tf.clip_by_value(sum_support,eps,float("inf")))

        dot_prod=tf.keras.backend.batch_dot(
            tf.expand_dims(query,1),tf.expand_dims(support_image,2))

        dot_prod=tf.squeeze(dot_prod)
        cos_sim=tf.multiply(dot_prod,support_magnitude)
        cos_sim=tf.reshape(cos_sim,[1,-1])
        similarities=tf.tensor_scatter_nd_update(similarities,[[i_sample]],cos_sim)
        i_sample=i_sample+1
    return tf.transpose(similarities)
def loss_acc(support_set,query_set,support_labels_onehot,query_labels_onehot):
    emb_imgs=[]
    for i in range(support_samples):
        emb_imgs.append(model_cnn(support_set[:,i,:,:,:]))
    for i_query in range(query_samples):
        outputs=model_lstm(query_set[:,i_query,:,:,:])
        query_emb=model_cnn(outputs)
        emb_imgs.append(query_emb)
        outputs=tf.stack(emb_imgs)

        similarities=cosine_similarity(outputs[:-1],outputs[-1])

        similarities=tf.nn.softmax(similarities)
        similarities=tf.expand_dims(similarities,1)
        preds=tf.squeeze(tf.keras.backend.batch_dot(similarities,support_labels_onehot))

        query_labels_new=query_labels_onehot[:,i_query,:]
        eq=tf.cast(tf.equal(tf.cast(tf.argmax(preds,axis=-1),tf.int32),
                          tf.cast(query_labels[:,i_query],tf.int32)),tf.float32)
        if i_query==0:
            loss=tf.keras.backend.categorical_crossentropy(query_labels_new,preds)
            acc=tf.reduce_mean(eq)
        else:
            loss=loss+tf.keras.backend.categorical_crossentropy(query_labels_new, preds)
            acc=acc+tf.reduce_mean(eq)
        del(emb_imgs[-1])
    loss=tf.keras.backend.mean(loss/query_samples)
    acc=acc/query_samples
    return loss,acc,model_cnn.trainable_variables,model_lstm.trainable_variables
support_samples=num_way*num_shot   #支持集样本个数
query_samples=num_way*num_query
optimizer=tf.keras.optimizers.Adam()
def train_step(suport,query,support_labels_onehot,query_labels_onehot):
    with tf.GradientTape() as t:
        loss_step,acc,cnn_trainable_variables,lstm_trainable_variables=loss_acc(
            suport,query,support_labels_onehot,query_labels_onehot)
    trainable_variables=cnn_trainable_variables+lstm_trainable_variables
    grads=t.gradient(loss_step,trainable_variables)
    optimizer.apply_gradients(zip(grads,trainable_variables))
    return loss_step,acc
num=10
epochs=100
train_loss=[]
train_acc=[]
for i in range(num):
    for epoch in range(epochs):
        support,query,support_labels,query_labels=few_shot(train_dataset)
        support_labels_onehot=tf.keras.utils.to_categorical(support_labels,
                                                            num_classes=num_way)
        query_labels_onehot=tf.keras.utils.to_categorical(query_labels,num_classes=num_way)
        epoch_loss,epoch_acc=train_step(
            support,query,support_labels_onehot,query_labels_onehot)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        print('.',end='')