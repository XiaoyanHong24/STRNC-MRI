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

path_test_0=glob.glob("dataset/T1/0/*.png")
display(path_test_0[0:3],len(path_test_0))
path_test_1=glob.glob("dataset/T1/1/*.png")
display(path_test_1[0:3],len(path_test_1))
path_test_2=glob.glob("dataset/T1/2/*.png")
display(path_test_2[0:3],len(path_test_2))
path_test_3=glob.glob("dataset/T1/3/*.png")
display(path_test_3[0:3],len(path_test_3))


train_num=4

#number of examples
num_examples=90

#image width
img_width=28

#image height
img_height=28
channels=1
train_dataset=np.zeros([train_num,num_examples,img_height,img_width,channels],
                       dtype=np.float32)
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
#测试数据
test_dataset=np.zeros([train_num,num_examples,img_height,img_width,channels],
                       dtype=np.float32)
test_dataset.shape
a=0
b=1
c=2
d=3
ran0=np.random.permutation(len(path_test_0))[:num_examples]
for index,name in enumerate(ran0):
    img_1=tf.io.read_file(path_test_0[name])
    img_1=tf.image.decode_png(img_1,channels=channels)
    img_1=tf.image.resize(img_1,[img_width,img_height])
    img_1=img_1/255
    test_dataset[a,index]=img_1

ran1=np.random.permutation(len(path_test_1))[:num_examples]
for index,name in enumerate(ran1):
    img_1=tf.io.read_file(path_test_1[name])
    img_1=tf.image.decode_png(img_1,channels=channels)
    img_1=tf.image.resize(img_1,[img_width,img_height])
    img_1=img_1/255
    test_dataset[b,index]=img_1

ran2=np.random.permutation(len(path_test_2))[:num_examples]
for index,name in enumerate(ran2):
    img_1=tf.io.read_file(path_test_2[name])
    img_1=tf.image.decode_png(img_1,channels=channels)
    img_1=tf.image.resize(img_1,[img_width,img_height])
    img_1=img_1/255
    test_dataset[c,index]=img_1

ran3=np.random.permutation(len(path_test_3))[:num_examples]
for index,name in enumerate(ran3):
    img_1=tf.io.read_file(path_test_3[name])
    img_1=tf.image.decode_png(img_1,channels=channels)
    img_1=tf.image.resize(img_1,[img_width,img_height])
    img_1=img_1/255
    test_dataset[d,index]=img_1
test_dataset.shape

num_examples=90
num_way=4

num_shot=1

num_query=15


batch=4
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

class STRNC(tf.keras.models.Model):
    def __init__(self, num_way):
        super(STRNC, self).__init__()
        self.Conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=7, padding='same',
                                            kernel_initializer='glorot_normal',
                                            activation='relu')
        self.Batch1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.Conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding='same',
                                            kernel_initializer='glorot_normal',
                                            activation='relu')
        self.Batch2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.Conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',
                                            kernel_initializer='glorot_normal',
                                            activation='relu')
        self.Batch3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.Conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='same',
                                            kernel_initializer='glorot_normal',
                                            activation='relu')
        self.Batch4 = tf.keras.layers.BatchNormalization(axis=-1)
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.fc = tf.keras.layers.Flatten()
        self.result = tf.keras.layers.Dense(num_way)
        
        # Adding LSTM layer
        self.model_lstm = tf.keras.Sequential()
        self.model_lstm.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)
        ))

    def call(self, inputs):
        # Pass through LSTM first
        x = self.model_lstm(inputs)
        
        # Pass through STRNC layers
        x = self.Conv1(x)
        x = self.Batch1(x, training=True)
        x = self.pool1(x)
        x = self.Conv2(x)
        x = self.Batch2(x, training=True)
        x = self.pool2(x)
        x = self.Conv3(x)
        x = self.Batch3(x, training=True)
        x = self.pool3(x)
        x = self.Conv4(x)
        x = self.Batch4(x, training=True)
        x = self.pool4(x)
        x = self.fc(x)
        x = self.result(x)
        return x

model=STRNC()
model.build(input_shape=(None,img_width,img_height,channels))
model.summary()
def meta_update(model, grad, alpha, num_way):
    copied_model = STRNC(num_way)
    copied_model.build(input_shape=(None, img_width, img_height, channels))
    
    copied_model.set_weights([w - alpha * g for w, g in zip(model.get_weights(), grad)])
    
    return copied_model

def inner_weights(model):
    layers_=[0,1,3,4,6,7,9,10,13]
    weights = model.get_weights()
    weights.append(model.model_lstm.get_weights())
    return weights
def model_to_copy(model):
    copied_model = STRNC(num_way)
    copied_model.build(input_shape=(None, img_width, img_height, channels))
    copied_model.set_weights(model.get_weights())
    copied_model.model_lstm.set_weights(model.model_lstm.get_weights())  # Copy LSTM weights
    return copied_model

def loss_(labels,pred):
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    result=loss(labels,pred)
    return result
def acc_(y_label,y_predict):
    epoch_acc=tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(y_label,axis=-1),tf.argmax(y_predict,axis=-1)),"float"))
    return epoch_acc

learning_rate_inner=1e-2
learning_rate_outer=1e-3
outer_optimizer=tf.keras.optimizers.Adam(learning_rate_outer)
inner_train_step=5
update_test_step=10
def train_step(support,query,support_labels_onehot,query_labels_onehot):
    train_loss=[]
    train_acc=[]
    with tf.GradientTape() as query_t:
        for index in range(batch):
            copy_model=model
            support_data=support[index]
            query_data=query[index]
            support_la=support_labels_onehot[index]
            query_la=query_labels_onehot[index]
            for inner_step in range(inner_train_step):
                with tf.GradientTape(watch_accessed_variables=False) as support_t:
                    support_t.watch(inner_weights(copy_model))
                    support_logits=copy_model(support_data)
                    support_loss=loss_(support_la,support_logits)
                inner_grads=support_t.gradient(support_loss,inner_weights(copy_model))
                copy_model=meta_update(copy_model,inner_grads,learning_rate_inner)

            query_logits=copy_model(query_data)
            query_pred=tf.nn.softmax(query_logits)
            query_loss=loss_(query_la,query_pred)

            epoch_acc=acc_(query_la,query_pred)

            train_loss.append(query_loss)
            train_acc.append(epoch_acc)
        meta_batch_loss=tf.reduce_mean(tf.stack(train_loss))

    outer_grads=query_t.gradient(meta_batch_loss,model.trainable_variables)
    outer_optimizer.apply_gradients(zip(outer_grads,model.trainable_variables))
    return meta_batch_loss,train_acc

def finetune_step(support,query,support_labels_onehot,query_labels_onehot):
    train_loss=[]
    train_acc=[]
    copy_model=model_to_copy(model)
    for index in range(batch):
        support_data=support[index]
        query_data=query[index]
        support_la=support_labels_onehot[index]
        query_la=query_labels_onehot[index]
        for inner_step in range(update_test_step):
            with tf.GradientTape(watch_accessed_variables=False) as test_tape:
                test_tape.watch(inner_weights(copy_model))
                support_logits=copy_model(support_data)
                support_loss=loss_(support_la,support_logits)
            inner_grads=test_tape.gradient(support_loss,inner_weights(copy_model))
            copy_model=meta_update(copy_model,inner_grads,learning_rate_inner)

        query_logits=copy_model(query_data)
        query_pred=tf.nn.softmax(query_logits)
        query_loss=loss_(query_la,query_pred)

        epoch_acc=acc_(query_la,query_pred)

        train_loss.append(query_loss)
        train_acc.append(epoch_acc)
    del copy_model
    return train_loss,train_acc
def eval_(model,support,query,support_labels_onehot,query_labels_onehot):
    train_loss=[]
    train_acc=[]
    for index in range(batch):
        support_data=support[index]
        query_data=query[index]
        support_la=support_labels_onehot[index]
        query_la=query_labels_onehot[index]

        query_logits=copy_model(query_data)
        query_pred=tf.nn.softmax(query_logits)
        query_loss=loss_(query_la,query_pred)
        epoch_acc=acc_(query_la,query_pred)

        train_loss.append(query_loss)
        train_acc.append(epoch_acc)
    every_loss=[loss.numpy() for loss in train_loss]
    every_acc=[acc.numpy() for acc in train_acc]
    print('loss={},acc={},loss={:.5f},acc_mean={:.5f}'.
                      format(every_loss,every_acc,np.mean(every_loss),np.mean(epoch_acc)))


num=5
epochs=10000
all_loss=[]
all_acc=[]
for i in range(num):
    for epoch in range(epochs):
        suport,query,support_labels,query_labels=few_shot(train_dataset)
        support_labels_onehot=tf.keras.utils.to_categorical(support_labels,num_way)
        query_labels_onehot=tf.keras.utils.to_categorical(query_labels,num_way)
        epoch_loss,epoch_acc=train_step(
            suport,query,support_labels_onehot,query_labels_onehot)
        every_acc=[acc.numpy() for acc in epoch_acc]
        all_loss.append(epoch_loss)
        all_acc.append(epoch_acc)
        if((epoch+1)%1==0):
            print('epoch={},{}task,loss={:.5f},acc={},acc_mean={:.5f}'.
                      format(i+1,epoch+1,epoch_loss,every_acc,np.mean(epoch_acc)))

        if((epoch+1)%10==0):
            suport,query,support_labels,query_labels=few_shot(test_dataset)
            support_labels_onehot=tf.keras.utils.to_categorical(support_labels,num_way)
            query_labels_onehot=tf.keras.utils.to_categorical(query_labels,num_way)
            epoch_loss,epoch_acc=finetune_step(
                suport,query,support_labels_onehot,query_labels_onehot)
            every_loss=[loss.numpy() for loss in epoch_loss]
            every_acc=[acc.numpy() for acc in epoch_acc]
            print('loss={},acc={}'.format(every_loss,every_acc,))
            print('loss={:.5f},acc_mean={:.5f}'.format(np.mean(
                every_loss),np.mean(epoch_acc)))