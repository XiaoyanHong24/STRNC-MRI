import numpy as np
import tensorflow as tf
from model import CNN, MetaLearner
from utils import load_dataset, generate_episode
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
def loss_acc(support_set,query_set,support_labels_onehot,query_labels_onehot):
    emb_imgs=[]
    for i in range(support_samples):
        emb_imgs.append(model_cnn(support_set[:,i,:,:,:]))
    #print(model_cnn.trainable_variables)
    for i_query in range(query_samples):
        query_emb=model_cnn(query_set[:,i_query,:,:,:])
        emb_imgs.append(query_emb)
        outputs=tf.stack(emb_imgs)

        # Fully contextual embedding
        outputs=model_lstm(outputs)

        # Cosine similarity between support set and query
        similarities=cosine_similarity(outputs[:-1],outputs[-1])

        # Produce predictions for target probabilities
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
        #emb_imgs.pop()
        #print(acc)
        del(emb_imgs[-1])
    #print(model_lstm.trainable_variables)
    loss=tf.keras.backend.mean(loss/query_samples)
    acc=acc/query_samples
    return loss,acc,model_cnn.trainable_variables,model_lstm.trainable_variables
# Configuration
config = {
    'img_size': 28,
    'num_way': 4,
    'num_shot': 1,
    'num_query': 1,
    'inner_steps': 5,
    'meta_lr': 1e-3,
    'inner_lr': 1e-2,
    'epochs': 10000,
    'batch_size': 4
}
def loss_(labels,pred):
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    result=loss(labels,pred)
    return result
def acc_(y_label,y_predict):
    epoch_acc=tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(y_label,axis=-1),tf.argmax(y_predict,axis=-1)),"float"))
    return epoch_acc
# Initialize model and optimizer
model_cnn = CNN(config['num_way'], config['img_size'])
meta_optimizer = tf.keras.optimizers.Adam(config['meta_lr'])
model_lstm=tf.keras.Sequential()
model_lstm.add(tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(64,return_sequences=True)))
# Load datasets
support_set,query_set,support_labels,query_labels=few_shot(train_dataset)
display(support_set.shape,query_set.shape,support_labels.shape,query_labels.shape)
support_samples=num_way*num_shot
query_samples=num_way*num_query
optimizer=tf.keras.optimizers.Adam()
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
num=10
epochs=100
train_loss=[]
train_acc=[]
# Training loop
for i in range(num):
    for epoch in range(epochs):
        suport,query,support_labels,query_labels=few_shot(train_dataset)
        support_labels_onehot=tf.keras.utils.to_categorical(support_labels,num_way)
        query_labels_onehot=tf.keras.utils.to_categorical(query_labels,num_way)
        epoch_loss,epoch_acc=train_step(
            suport,query,support_labels_onehot,query_labels_onehot)
        every_acc=[acc.numpy() for acc in epoch_acc]
        #print(epoch_loss,epoch_acc)
        all_loss.append(epoch_loss)
        all_acc.append(epoch_acc)
        #print('.',end='')
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
            print('---------------------------------------')
            print('loss={},acc={}'.format(every_loss,every_acc,))
            print('loss={:.5f},acc_mean={:.5f}'.format(np.mean(
                every_loss),np.mean(epoch_acc)))