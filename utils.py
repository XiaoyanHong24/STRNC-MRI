# utils.py
import numpy as np
import tensorflow as tf
import glob

def load_dataset(base_path, num_classes, num_examples, img_size):
    dataset = []
    for i in range(num_classes):
        paths = glob.glob(f"{base_path}/{i}/*.png")[:num_examples]
        class_images = []
        for path in paths:
            img = tf.io.read_file(path)
            img = tf.image.decode_png(img, channels=1)
            img = tf.image.resize(img, [img_size, img_size])
            img = img / 255.0
            class_images.append(img)
        dataset.append(np.array(class_images))
    return np.array(dataset)

def generate_episode(dataset, num_way, num_shot, num_query, batch_size):
    support = np.zeros([batch_size, num_way*num_shot, *dataset.shape[2:]])
    query = np.zeros([batch_size, num_way*num_query, *dataset.shape[2:]])
    s_labels = np.zeros([batch_size, num_way*num_shot], dtype=np.int32)
    q_labels = np.zeros([batch_size, num_way*num_query], dtype=np.int32)
    
    for i in range(batch_size):
        classes = np.random.permutation(len(dataset))[:num_way]
        for c_idx, c in enumerate(classes):
            samples = np.random.permutation(len(dataset[c]))[:num_shot+num_query]
            support[i, c_idx*num_shot:(c_idx+1)*num_shot] = dataset[c, samples[:num_shot]]
            query[i, c_idx*num_query:(c_idx+1)*num_query] = dataset[c, samples[num_shot:]]
            s_labels[i, c_idx*num_shot:(c_idx+1)*num_shot] = c
            q_labels[i, c_idx*num_query:(c_idx+1)*num_query] = c
            
    return support, query, s_labels, q_labels


