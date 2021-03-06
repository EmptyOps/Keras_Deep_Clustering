from keras.datasets import mnist
import numpy as np
np.random.seed(10)

from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
# from sklearn.cluster import KMeans
#import keras.initializers.MiniBatchKMeans as KMeans
from sklearn.cluster import MiniBatchKMeans as KMeans
import metrics
from DataGenerator import DataGenerator

import cv2
from random import randint
import os, sys
import shutil 

####################
import argparse

#use absolute paths
ABS_PATh = os.path.dirname(os.path.abspath(__file__)) + "/"

# Instantiate the parser
parser = argparse.ArgumentParser(description='a utility')

parser.add_argument('-d', '--dir_to_process', type=str, nargs='?', help='dir_to_process, separate by comma "," if multiple directories')
parser.add_argument('-o', '--out_to_dir',type=str, nargs='?',help='output will be written to out_to_dir ')
parser.add_argument('-b', '--is_debug', action='store_true', help='A boolean True False')
parser.add_argument('-s', '--is_use_sample_data', action='store_true', help='A boolean True False')

#for embedding models 
parser.add_argument('--is_apply_text_preprocessing', action='store_true', help='A boolean True False')
parser.add_argument('--is_apply_sequence_preprocessing', action='store_true', help='A boolean True False')


parser.add_argument('--batch_size', type=int, nargs='?', help='batch_size')
parser.add_argument('--n_dim_1', type=int, nargs='?', help='n_classes')
parser.add_argument('--n_dim_2', type=int, nargs='?', help='n_classes')
parser.add_argument('--n_channels', type=int, nargs='?', help='n_channels')
parser.add_argument('--n_classes', type=int, nargs='?', help='n_classes')

parser.add_argument('-pt', '--pretrain_epochs', type=str, nargs='?', help='pretrain_epochs')
parser.add_argument('-mi', '--maxiter', type=str, nargs='?', help='maxiter')
parser.add_argument('-ui', '--update_interval', type=str, nargs='?', help='update_interval')

parser.add_argument('-cmsp', '--confusion_matrix_save_path', type=str, nargs='?', help='confusion_matrix_save_path')

FLAGS = parser.parse_args()
print(FLAGS)

if FLAGS.dir_to_process == "":
    paths = []  #specify static here
else:
    if not "," in FLAGS.dir_to_process:
        paths = [FLAGS.dir_to_process+"/" ]
    else:
        paths = FLAGS.dir_to_process.split(",")

if FLAGS.out_to_dir == None or FLAGS.out_to_dir == "":
    raise Exception("Please specify out_to_dir")
else:
    if not os.path.exists( FLAGS.out_to_dir ):
        os.mkdir( FLAGS.out_to_dir )

####################    


def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    print( "autoencoder" )
    print(dims)
    n_stacks = len(dims) - 1
    # input
    input_img = Input(shape=(dims[0],), name='input')
    x = input_img
    # internal layers in encoder
    for i in range(n_stacks-1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

    # hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

    print(x.shape)
    # output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')


y_paths = []
if FLAGS.is_use_sample_data:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)

    # print(x)    
    # print(y)
    print( len(x) )     
    print( len(y) ) 
    print( x.shape )     
    print( y.shape ) 
    print( x[0].max(axis=0) )

    n_clusters = len(np.unique(y))
    print( x.shape )
    print( y.shape )
    
    raise Exception("Mnist sample not supported yet with data generator")

else:
    data = DataGenerator(list_IDs=None, labels=None, batch_size=FLAGS.batch_size, dim=(int(FLAGS.n_dim_1), int(FLAGS.n_dim_2)), n_channels=int(FLAGS.n_channels), 
                 n_classes=FLAGS.n_classes, shuffle=True, datatype='imgs_to_gray', datadirs=paths, 
                 is_label_to_categorical=False, is_normalize_image_datatype=True, is_apply_text_preprocessing=FLAGS.is_apply_text_preprocessing, 
                 is_apply_sequence_preprocessing = FLAGS.is_apply_sequence_preprocessing)

    n_clusters = FLAGS.n_classes

print( "n_clusters" )
print( n_clusters )
kmeans = KMeans(n_clusters=n_clusters, n_init=20, batch_size=FLAGS.batch_size)   #random_state=0,    #, n_jobs=4)
y_pred_kmeans = kmeans
if True:     #False:   #ToDo temp
    for bi in range(0, data.__len__()):
        x, y = data.__getitem__(bi, True, is_return_only_x=False)
        y_pred_kmeans = y_pred_kmeans.partial_fit(x[:,:])
        
        ##
        print("y ", y.shape, y_pred_kmeans.labels_.shape)
        #print( metrics.acc(y, y_pred_kmeans.labels_) )
else:    
    x, y = data.__getitem__(0, True, is_return_only_x=False)

#to speed up moved out side loop, will it going to change anything with efficiency of the algorithm 
#print( metrics.acc(y, y_pred_kmeans.labels_) ) #padd entire y


#dims = [x.shape[-1], 500, 500, 2000, 10]
#dims = [x.shape[-1], 500, 500, 600, 10]
dims = [x.shape[-1], 500, 500, 600, 234]
init = VarianceScaling(scale=1. / 3., mode='fan_in',
                           distribution='uniform')
pretrain_optimizer = SGD(lr=1, momentum=0.9)
pretrain_epochs = int(FLAGS.pretrain_epochs) #1200  # 300
batch_size = FLAGS.batch_size   # 256
save_dir = './results'

autoencoder, encoder = autoencoder(dims, init=init)

from keras.utils import plot_model
from IPython.display import Image
plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)
Image(filename='autoencoder.png') 
plot_model(encoder, to_file='encoder.png', show_shapes=True)
Image(filename='encoder.png') 

autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
#autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs) #, callbacks=cb)
autoencoder.fit_generator(generator=data, epochs=pretrain_epochs) #, callbacks=cb)
autoencoder.save_weights(save_dir + '/ae_weights.h5')
autoencoder.save_weights(save_dir + '/ae_weights.h5')
autoencoder.load_weights(save_dir + '/ae_weights.h5')

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input, outputs=clustering_layer)

from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)
from IPython.display import Image
Image(filename='model.png') 


model.compile(optimizer=SGD(0.01, 0.9), loss='kld')

kmeans = KMeans(n_clusters=n_clusters, n_init=20, batch_size=FLAGS.batch_size)   # random_state=0,  #KMeans(n_clusters=n_clusters, n_init=20)
#y_pred = kmeans.fit_predict(encoder.predict(x))
is_first = True
y_pred = None
for bi in range(0, data.__len__()):
    x, y = data.__getitem__(bi, True, is_return_only_x=False)
    # if is_first:
    #     is_first = False
    #     y_pred = kmeans.fit_predict(x[:,:])
    # else:
    #     y_pred = np.concatenate(y_pred, kmeans.fit_predict(x[:,:]))
    kmeans.fit_predict(x[:,:])

y_pred_last = kmeans.labels_     #np.copy(y_pred)
print( "y_pred_last" )
print( type(y_pred_last) )
print( y_pred_last.shape )
print( "kmeans.cluster_centers_ .........................................................................................." )
print( n_clusters )
print(  type(kmeans.cluster_centers_) )
print( kmeans.cluster_centers_.shape )

model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

loss = 0
index = 0
maxiter = int(FLAGS.maxiter) #32000 # 8000
update_interval = int(FLAGS.update_interval) #140
# index_array = np.arange(x.shape[0])

tol = 0.001 # tolerance threshold to stop training

yALL = None
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        #q = model.predict(x, verbose=0)
        q = None
        for bi in range(0, data.__len__()):
            x, y = data.__getitem__(bi, True, is_return_only_x=False)
            if q is None:
                q = model.predict(x[:,:], verbose=0)
                yALL = y
            else:
                print( type(q), q.shape ) 
                qt = model.predict(x[:,:], verbose=0 )
                print( type(qt), qt.shape ) 
                q = np.concatenate( ( q, qt ), axis = 0 )
                print( type(y), yALL.shape, y.shape ) 
                yALL = np.concatenate( (yALL, y), axis = 0 )

        p = target_distribution(q)  # update the auxiliary target distribution p

        # evaluate the clustering performance
        y_pred = q.argmax(1)
        if y is not None:
            #ToDo to set accuracy need to compare with yALL
            acc = 0 # np.round(metrics.acc(y, y_pred), 5)
            nmi = 0 # np.round(metrics.nmi(y, y_pred), 5)
            ari = 0 # np.round(metrics.ari(y, y_pred), 5)
            loss = np.round(loss, 5)
            print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

        # check stop criterion - model convergence
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    # idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
    # loss = model.train_on_batch(x=x[idx], y=p[idx])
    # index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
    x, y = data.__getitem__(index, True, is_return_only_x=False)
    loss = model.train_on_batch(x=x, y=p[ index * FLAGS.batch_size : (index+1) * FLAGS.batch_size ])      #y=p[idx])
    index = index + 1 if (index + 1) < data.__len__() else 0

model.save_weights(save_dir + '/DEC_model_final.h5')

model.load_weights(save_dir + '/DEC_model_final.h5')

# Eval.
#q = model.predict(x, verbose=0)
q = None
for bi in range(0, data.__len__()):
    x, y = data.__getitem__(bi, True, is_return_only_x=False, is_keep_shuffled_index = True)
    if q is None:
        q = model.predict(x[:,:], verbose=0)
    else:
        q = np.concatenate( ( q, model.predict(x[:,:], verbose=0) ), axis = 0 )
p = target_distribution(q)  # update the auxiliary target distribution p

# evaluate the clustering performance
y_pred = q.argmax(1)
if y is not None:
    acc = 0 #np.round(metrics.acc(y, y_pred), 5)
    nmi = 0 #np.round(metrics.nmi(y, y_pred), 5)
    ari = 0 #np.round(metrics.ari(y, y_pred), 5)
    loss = np.round(loss, 5)
    print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)

import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt
sns.set(font_scale=3)
print( "yALL ", type(yALL), type(y_pred), yALL.shape, y_pred.shape )
confusion_matrix = sklearn.metrics.confusion_matrix(yALL, y_pred)  #(y, y_pred)

#label
if not FLAGS.out_to_dir == None and not FLAGS.out_to_dir == "":
    bi = 0
    y_paths = None
    sizeyp = len(y_pred)
    for i in range(0, sizeyp):
        imd = np.mod(i, FLAGS.batch_size)
        if imd == 0:
            y_paths = data.__getitem__(bi, True, is_return_only_x=False, is_return_last_paths = True)
            bi += 1

        if not os.path.isdir( FLAGS.out_to_dir + "/" + str(y_pred[i]) ):
            os.mkdir( FLAGS.out_to_dir + "/" + str(y_pred[i]) )

        #ToDo
        #os.rename( y_paths[i], FLAGS.out_to_dir + "/" + str(y_pred[i]) + "/" + os.path.basename(y_paths[i]) )
        shutil.copyfile(y_paths[ imd ], FLAGS.out_to_dir + "/" + str(y_pred[i]) + "/" + os.path.basename(y_paths[imd])) 


plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
if not FLAGS.confusion_matrix_save_path == None and not FLAGS.confusion_matrix_save_path == "":
    plt.savefig(FLAGS.confusion_matrix_save_path)
else:
    plt.show()

input("Confusion matrix created, press enter to continue further...")


from sklearn.utils.linear_assignment_ import linear_assignment

y_true = y.astype(np.int64)
D = max(y_pred.max(), y_true.max()) + 1
w = np.zeros((D, D), dtype=np.int64)
# Confusion matrix.
for i in range(y_pred.size):
    w[y_pred[i], y_true[i]] += 1
ind = linear_assignment(-w)

sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

w

ind

w.argmax(1)

from keras.models import Model
from keras import backend as K
from keras import layers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Conv2DTranspose
from keras.models import Model
import numpy as np

def autoencoderConv2D_1(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):
    input_img = Input(shape=input_shape)
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    x = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape)(input_img)

    x = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2')(x)

    x = Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3')(x)

    x = Flatten()(x)
    encoded = Dense(units=filters[3], name='embedding')(x)
    x = Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu')(encoded)

    x = Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2]))(x)
    x = Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3')(x)

    x = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2')(x)

    decoded = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1')(x)
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

def autoencoderConv2D_2(img_shape=(28, 28, 1)):
    """
    Conv2D auto-encoder model.
    Arguments:
        img_shape: e.g. (28, 28, 1) for MNIST
    return:
        (autoencoder, encoder), Model of autoencoder and model of encoder
    """
    input_img = Input(shape=img_shape)
    # Encoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same', strides=(2, 2))(input_img)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
    shape_before_flattening = K.int_shape(x)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = Flatten()(x)
    encoded = Dense(10, activation='relu', name='encoded')(x)

    # Decoder
    x = Dense(np.prod(shape_before_flattening[1:]),
                activation='relu')(encoded)
    # Reshape into an image of the same shape as before our last `Flatten` layer
    x = Reshape(shape_before_flattening[1:])(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')


autoencoder, encoder = autoencoderConv2D_1()

autoencoder.summary()

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
x = x.reshape(x.shape + (1,))
x = np.divide(x, 255.)

pretrain_epochs = 100
batch_size = 256

autoencoder.compile(optimizer='adadelta', loss='mse')

autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs)
autoencoder.save_weights(save_dir+'/conv_ae_weights.h5')

autoencoder.load_weights(save_dir+'/conv_ae_weights.h5')

clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input, outputs=clustering_layer)
model.compile(optimizer='adam', loss='kld')

kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(x))

y_pred_last = np.copy(y_pred)

model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

loss = 0
index = 0
maxiter = 8000
update_interval = 140
index_array = np.arange(x.shape[0])

tol = 0.001 # tolerance threshold to stop training

for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model.predict(x, verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p

        # evaluate the clustering performance
        y_pred = q.argmax(1)
        if y is not None:
            acc = np.round(metrics.acc(y, y_pred), 5)
            nmi = np.round(metrics.nmi(y, y_pred), 5)
            ari = np.round(metrics.ari(y, y_pred), 5)
            loss = np.round(loss, 5)
            print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

        # check stop criterion
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
    loss = model.train_on_batch(x=x[idx], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

model.save_weights(save_dir + '/conv_DEC_model_final.h5')

model.load_weights(save_dir + '/conv_DEC_model_final.h5')

# Eval.
q = model.predict(x, verbose=0)
p = target_distribution(q)  # update the auxiliary target distribution p

# evaluate the clustering performance
y_pred = q.argmax(1)
if y is not None:
    acc = np.round(metrics.acc(y, y_pred), 5)
    nmi = np.round(metrics.nmi(y, y_pred), 5)
    ari = np.round(metrics.ari(y, y_pred), 5)
    loss = np.round(loss, 5)
    print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)

import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt
sns.set(font_scale=3)
confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)

plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()

autoencoder, encoder = autoencoderConv2D_1()
autoencoder.load_weights(save_dir+'/conv_ae_weights.h5')
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input,
                           outputs=[clustering_layer, autoencoder.output])

from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)
from IPython.display import Image
Image(filename='model.png') 


kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(x))
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
y_pred_last = np.copy(y_pred)

model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer='adam')

for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q, _  = model.predict(x, verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p

        # evaluate the clustering performance
        y_pred = q.argmax(1)
        if y is not None:
            acc = np.round(metrics.acc(y, y_pred), 5)
            nmi = np.round(metrics.nmi(y, y_pred), 5)
            ari = np.round(metrics.ari(y, y_pred), 5)
            loss = np.round(loss, 5)
            print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

        # check stop criterion
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
    loss = model.train_on_batch(x=x[idx], y=[p[idx], x[idx]])
    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

model.save_weights(save_dir + '/conv_b_DEC_model_final.h5')

model.load_weights(save_dir + '/conv_b_DEC_model_final.h5')

# Eval.
q, _ = model.predict(x, verbose=0)
p = target_distribution(q)  # update the auxiliary target distribution p

# evaluate the clustering performance
y_pred = q.argmax(1)
if y is not None:
    acc = np.round(metrics.acc(y, y_pred), 5)
    nmi = np.round(metrics.nmi(y, y_pred), 5)
    ari = np.round(metrics.ari(y, y_pred), 5)
    loss = np.round(loss, 5)
    print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)

import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt
sns.set(font_scale=3)
confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)

plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
x = x.reshape((x.shape[0], -1))
x = np.divide(x, 255.)
n_clusters = len(np.unique(y))
x.shape

dims = [x.shape[-1], 500, 500, 2000, 10]
init = VarianceScaling(scale=1. / 3., mode='fan_in',
                           distribution='uniform')
pretrain_optimizer = SGD(lr=1, momentum=0.9)
pretrain_epochs = 300
batch_size = 256
save_dir = './results'


# def autoencoder(dims, act='relu', init='glorot_uniform'):
#     """
#     Fully connected auto-encoder model, symmetric.
#     Arguments:
#         dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
#             The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
#         act: activation, not applied to Input, Hidden and Output layers
#     return:
#         (ae_model, encoder_model), Model of autoencoder and model of encoder
#     """
#     n_stacks = len(dims) - 1
#     # input
#     input_img = Input(shape=(dims[0],), name='input')
#     x = input_img
#     # internal layers in encoder
#     for i in range(n_stacks-1):
#         x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

#     # hidden layer
#     encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

#     x = encoded
#     # internal layers in decoder
#     for i in range(n_stacks-1, 0, -1):
#         x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

#     # output
#     x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
#     decoded = x
#     return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')


autoencoder, encoder = autoencoder(dims, init=init)
autoencoder.load_weights(save_dir+'/ae_weights.h5')
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input,
            outputs=[clustering_layer, autoencoder.output])

from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)
from IPython.display import Image
Image(filename='model.png') 

kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(x))
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
y_pred_last = np.copy(y_pred)

model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer=pretrain_optimizer)

for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q, _  = model.predict(x, verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p

        # evaluate the clustering performance
        y_pred = q.argmax(1)
        if y is not None:
            acc = np.round(metrics.acc(y, y_pred), 5)
            nmi = np.round(metrics.nmi(y, y_pred), 5)
            ari = np.round(metrics.ari(y, y_pred), 5)
            loss = np.round(loss, 5)
            print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

        # check stop criterion
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
    loss = model.train_on_batch(x=x[idx], y=[p[idx], x[idx]])
    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

model.save_weights(save_dir + '/b_DEC_model_final.h5')

model.load_weights(save_dir + '/b_DEC_model_final.h5')

# Eval.
q, _ = model.predict(x, verbose=0)
p = target_distribution(q)  # update the auxiliary target distribution p

# evaluate the clustering performance
y_pred = q.argmax(1)
if y is not None:
    acc = np.round(metrics.acc(y, y_pred), 5)
    nmi = np.round(metrics.nmi(y, y_pred), 5)
    ari = np.round(metrics.ari(y, y_pred), 5)
    loss = np.round(loss, 5)
    print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)

import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt
sns.set(font_scale=3)
confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)

plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()


