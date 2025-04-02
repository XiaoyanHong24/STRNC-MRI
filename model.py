import tensorflow as tf
def meta_update(model,grad,alpha):
    layers_=[0,1,3,4,6,7,9,10,13]
    copied_model=model_1()
    copied_model.build(input_shape=(None,img_width,img_height,channels))

    copied_model.Conv1.kernel=model.Conv1.kernel
    copied_model.Conv1.bias=model.Conv1.bias
    copied_model.Batch1.gamma=model.Batch1.gamma
    copied_model.Batch1.beta=model.Batch1.beta

    copied_model.Conv2.kernel=model.Conv2.kernel
    copied_model.Conv2.bias=model.Conv2.bias
    copied_model.Batch2.gamma=model.Batch2.gamma
    copied_model.Batch2.beta=model.Batch2.beta

    copied_model.Conv3.kernel=model.Conv3.kernel
    copied_model.Conv3.bias=model.Conv3.bias
    copied_model.Batch3.gamma=model.Batch3.gamma
    copied_model.Batch3.beta=model.Batch3.beta

    copied_model.Conv4.kernel=model.Conv4.kernel
    copied_model.Conv4.bias=model.Conv4.bias
    copied_model.Batch4.gamma=model.Batch4.gamma
    copied_model.Batch4.beta=model.Batch4.beta

    copied_model.result.kernel=model.result.kernel
    copied_model.result.bias=model.result.bias


    copied_model.Conv1.kernel.assign_sub(alpha*grad[0])
    copied_model.Conv1.bias.assign_sub(alpha*grad[1])
    copied_model.Batch1.gamma.assign_sub(alpha*grad[2])
    copied_model.Batch1.beta.assign_sub(alpha*grad[3])


    copied_model.Conv2.kernel.assign_sub(alpha*grad[4])
    copied_model.Conv2.bias.assign_sub(alpha*grad[5])
    copied_model.Batch2.gamma.assign_sub(alpha*grad[6])
    copied_model.Batch2.beta.assign_sub(alpha*grad[7])

    copied_model.Conv3.kernel.assign_sub(alpha*grad[8])
    copied_model.Conv3.bias.assign_sub(alpha*grad[9])

    copied_model.Batch3.gamma.assign_sub(alpha*grad[10])
    copied_model.Batch3.beta.assign_sub(alpha*grad[11])


    copied_model.Conv4.kernel.assign_sub(alpha*grad[12])
    copied_model.Conv4.bias.assign_sub(alpha*grad[13])
    copied_model.Batch4.gamma.assign_sub(alpha*grad[14])
    copied_model.Batch4.beta.assign_sub(alpha*grad[15])

    copied_model.result.kernel.assign_sub(alpha*grad[16])
    copied_model.result.bias.assign_sub(alpha*grad[17])

    return copied_model

def inner_weights(model):
    layers_=[0,1,3,4,6,7,9,10,13]
    #layers_=[1,2,4,5,7,8,10,11,14]
    weights=[
        model.Conv1.kernel,
        model.Conv1.bias,
        model.Batch1.gamma,
        model.Batch1.beta,
        model.Conv2.kernel,
        model.Conv2.bias,
        model.Batch2.gamma,
        model.Batch2.beta,
        model.Conv3.kernel,
        model.Conv3.bias,
        model.Batch3.gamma,
        model.Batch3.beta,
        model.Conv4.kernel,
        model.Conv4.bias,
        model.Batch4.gamma,
        model.Batch4.beta,
        model.result.kernel,
        model.result.bias,

    ]
    return weights
def model_to_copy(model):
    copied_model=model_1()
    copied_model.build(input_shape=(None,img_width,img_height,channels))

    copied_model.Conv1.kernel=model.Conv1.kernel
    copied_model.Conv1.bias=model.Conv1.bias
    copied_model.Batch1.gamma=model.Batch1.gamma
    copied_model.Batch1.beta=model.Batch1.beta

    copied_model.Conv2.kernel=model.Conv2.kernel
    copied_model.Conv2.bias=model.Conv2.bias
    copied_model.Batch2.gamma=model.Batch2.gamma
    copied_model.Batch2.beta=model.Batch2.beta

    copied_model.Conv3.kernel=model.Conv3.kernel
    copied_model.Conv3.bias=model.Conv3.bias
    copied_model.Batch3.gamma=model.Batch3.gamma
    copied_model.Batch3.beta=model.Batch3.beta

    copied_model.Conv4.kernel=model.Conv4.kernel
    copied_model.Conv4.bias=model.Conv4.bias
    copied_model.Batch4.gamma=model.Batch4.gamma
    copied_model.Batch4.beta=model.Batch4.beta

    copied_model.result.kernel=model.result.kernel
    copied_model.result.bias=model.result.bias
    return copied_model
class model_1(tf.keras.Model):
    def __init__(self, num_way, img_size=28):
        super(model_1, self).__init__()
        self.conv_layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_way)
        ])
        
    def call(self, inputs):
        return self.conv_layers(inputs)


class model_2(tf.keras.Model):
    def __init__(self, num_way, img_size=28):
        super(model_2, self).__init__()
        self.conv_layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 7, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(64, 5, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(64, 1, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_way)
        ])
        
    def call(self, inputs):
        return self.conv_layers(inputs)
class model_3(tf.keras.models.Model):
    def __init__(self):
        super(model_3,self).__init__()
        self.Conv1=tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',
                                          kernel_initializer='glorot_normal',
                                 activation='relu')
        self.Batch1=tf.keras.layers.BatchNormalization(axis=-1)
        self.pool1=tf.keras.layers.MaxPool2D(pool_size=(2,2))
        self.Conv2=tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',
                                 kernel_initializer='glorot_normal',
                                 activation='relu')
        self.Batch2=tf.keras.layers.BatchNormalization(axis=-1)
        self.pool2=tf.keras.layers.MaxPool2D(pool_size=(2,2))

        self.Conv3=tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',
                                 kernel_initializer='glorot_normal',
                                 activation='relu')
        self.Batch3=tf.keras.layers.BatchNormalization(axis=-1)
        self.pool3=tf.keras.layers.MaxPool2D(pool_size=(2,2))

        self.Conv4=tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',
                                 kernel_initializer='glorot_normal',
                                 activation='relu')
        self.Batch4=tf.keras.layers.BatchNormalization(axis=-1)
        self.pool4=tf.keras.layers.MaxPool2D(pool_size=(2,2))

        self.fc=tf.keras.layers.Flatten()
        #self.result=tf.keras.layers.Dense(5,activation='softmax')
        self.result=tf.keras.layers.Dense(num_way)
    def call(self,inputs):
        x=self.Conv1(inputs)
        x=self.Batch1(x,training=True)
        x=self.pool1(x)
        x=self.Conv2(x)
        x=self.Batch2(x,training=True)
        x=self.pool2(x)
        x=self.Conv3(x)
        x=self.Batch3(x,training=True)
        x=self.pool3(x)
        x=self.Conv4(x)
        x=self.Batch4(x,training=True)
        x=self.pool4(x)
        x=self.fc(x)
        x=self.result(x)
        return x

class MetaLearner:
    @staticmethod
    def meta_update(model, grads, alpha):
        variables = model.trainable_variables
        for var, grad in zip(variables, grads):
            if grad is not None:
                var.assign_sub(alpha * grad)
        return model