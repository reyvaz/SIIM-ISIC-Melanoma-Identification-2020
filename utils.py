import math, re
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
import matplotlib as mpl
from matplotlib import pyplot as plt

# EfficientNet Version-Image Size Dictionary
efn_dict = {0: {'model': efn.EfficientNetB0, 'size': 224},
            1: {'model': efn.EfficientNetB1, 'size': 240},
            2: {'model': efn.EfficientNetB2, 'size': 260},
            3: {'model': efn.EfficientNetB3, 'size': 300},
            4: {'model': efn.EfficientNetB4, 'size': 380},
            5: {'model': efn.EfficientNetB5, 'size': 456},
            6: {'model': efn.EfficientNetB6, 'size': 528},
            7: {'model': efn.EfficientNetB7, 'size': 600}}

# Counting Utils
def count_data_items(filenames):
    # Counts data items in a list of TFRecs
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

def trainable_parameter_count(model):
    # Breaks down parameter counts in a tf.keras or keras model.
    trainable_count = np.int(np.sum([K.count_params(w) for w in model.trainable_weights]))
    non_trainable_count = np.int(np.sum([K.count_params(w) for w in model.non_trainable_weights]))
    total_count = trainable_count + non_trainable_count

    print('Total params: {:,}'.format(total_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))
    return total_count, trainable_count, non_trainable_count


#Augmentation Utils: Shear, Rotate, Zoom-in
# Credit: shear_img() and rotate_image() adapted from Chris Deotte's code found here:
# https://www.kaggle.com/cdeotte/triple-stratified-kfold-with-tfrecords/data#Step-3:-Build-Model

@tf.function
def rotate_img(image, DIM, rotate_factor = 45.0):
    # image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated
    XDIM = DIM%2 #fix for odd size

    rot = rotate_factor * tf.random.normal([1], dtype='float32')
    rotation = math.pi * rot / 180. # degrees to radians

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])

    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    m = get_3x3_mat([c1,   s1,   zero,
                     -s1,  c1,   zero,
                     zero, zero, one])

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)
    y = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
    z = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack( [x,y,z] )
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)
    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    return tf.reshape(d,[DIM, DIM,3])

@tf.function
def shear_img(image, DIM, shear_factor = 7.0):
    # image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - randomly sheared image
    XDIM = DIM%2 #fix for odd size

    shr = shear_factor * tf.random.normal([1], dtype='float32')
    shear    = math.pi * shr    / 180. # degrees to radians

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    m = get_3x3_mat([one,  s2,   zero,
                     zero, c2,   zero,
                     zero, zero, one])
    # LIST DESTINATION PIXEL INDICES
    x   = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)
    y   = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
    z   = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack( [x,y,z] )
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)
    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
    d    = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d,[DIM, DIM,3])

@tf.function
def central_zoom(image, resize, zoom_factor = 0.6):
    img = tf.image.central_crop(image, zoom_factor)
    img = tf.image.resize(img, resize)
    return img

# Plot Utils
# matplotlib plotting parameters
axes_color = '#999999'
mpl.rcParams.update({'text.color' : "#999999", 'axes.labelcolor' : axes_color,
                     'font.size': 10, 'xtick.color':axes_color,'ytick.color':axes_color,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': axes_color, 'axes.linewidth':1.0, 'figure.figsize':[8, 4]})

def generate_examples(ds, n_batches = 5):
    # will generate [image, label] pairs from ds
    # each batch contain BATCH_SIZE pairs
    ds = ds.take(n_batches)
    dataset_tuples = []
    for i, (image, label) in enumerate(ds):
        for j in range(len(image)):
            dataset_tuples.append([image[j], label[j]])
    return dataset_tuples

CLASSES = ['Benign', 'Malignant']
def plot_example(dataset_pair, print_size=False):
    input_image = dataset_pair[0]
    label = dataset_pair[1].numpy()
    print('Label: {}'.format(CLASSES[label]))

    if print_size:
        print('Image size: {}'.format(input_image.shape))

    plt.figure(figsize=(3.5, 3.5))
    plt.imshow(tf.keras.preprocessing.image.array_to_img(input_image))
    plt.axis('off')
    plt.show()
    return None

def plot_image(input_image, print_size=True):
    if print_size:
        print('Image size: {}'.format(input_image.shape))
    plt.figure(figsize=(3.5, 3.5))
    plt.imshow(tf.keras.preprocessing.image.array_to_img(input_image))
    plt.axis('off')
    plt.show()
    return None

def plot_lr_timeline(lrfn, lr_params, num_epochs = 20, show_list=False):
    lr_timeline = [lrfn(i, lr_params) for i in range(20)]
    plt.plot(lr_timeline)
    plt.show()
    if show_list: print(lr_timeline)

# Train Utils
def config_checkpoint(filepath = 'weights.h5', monitor ='val_auc', mode = 'max'):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath = filepath,
        monitor = monitor,
        mode = mode,
        save_best_only = True,
        save_weights_only=True,
        verbose = 0)
    return checkpoint

LEARNING_RATE = 3e-4
opts = {'Nadam': tf.keras.optimizers.Nadam(learning_rate=LEARNING_RATE),
        'Radam': tfa.optimizers.RectifiedAdam(learning_rate=LEARNING_RATE),
        'Adam': tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        'SGD': tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)}


# def get_optimizer(opt, lr):
#     LEARNING_RATE = lr
#     optimizers = {'Nadam': tf.keras.optimizers.Nadam(learning_rate=LEARNING_RATE),
#                   'Radam': tfa.optimizers.RectifiedAdam(learning_rate=LEARNING_RATE),
#                   'Adam': tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
#                   'SGD': tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)}
#     return optimizers[opt]

# def describe_ds(ds):
#     print(str(ds).replace('<', '').replace('>', ''))
def describe_ds(ds):
    print(str(ds).replace('<PrefetchDataset', 'Dataset:').replace('>', ''))
