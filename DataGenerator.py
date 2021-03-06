import numpy as np
import keras

import cv2
from random import randint
import os, sys

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs=None, labels=None, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, datatype='.npy', datadirs=[], is_label_to_categorical=False, is_normalize_image_datatype=False, 
                 is_apply_text_preprocessing = False, is_apply_sequence_preprocessing = False):
        'Initialization need to provide either list_IDs or datadirs(dir paths)'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.shuffle_cache = None
        self.datatype = datatype
        self.datadirs = datadirs
        self.is_label_to_categorical = is_label_to_categorical
        self.is_normalize_image_datatype = is_normalize_image_datatype


        if not list_IDs == None:
            self.is_dir_based_data_batches = False
            self.list_IDs = list_IDs
        else:
            self.is_dir_based_data_batches = True
            self.dir_batch_size = self.batch_size * 50
            self.__init_IDs__(list_IDs, datadirs)            

        if not labels == None:
            self.labels = labels
        else:
            if not list_IDs == None:    #for datadirs the labels will be generated at runtime on first load and then updated and maintained on memory for the remaining runtime
                self.labels = self.init_random_labels( len(list_IDs), n_classes)

        self.on_epoch_end( is_xplicit_call = True )

    def __init_IDs__(self, list_IDs, datadirs):
        'usefull when number of records is large, to make it easier to even load ids by batch'

        # support multiple dirs to make it easier to train large dataset in different dirs without moving them however to offset the gradient leaning towards 
        # one direction for one dir and for the other on the other direction it is made sure that large enough indexes are loaded together(to keep benefits 
        # of shuffeling) like loaded_indexes = batch_size * 50
        self.datadirs = datadirs
        self.datadirs_rec_count = []

        #count total
        self.total_records = 0
        sdir = len(datadirs)
        for i in range(0, sdir):
            print( "__init_IDs__", self.datadirs[i] )
            self.datadirs_rec_count.append( len([True for name in os.listdir(self.datadirs[i]) if os.path.isfile( os.path.join( self.datadirs[i], name ) ) ] ) )
            self.total_records = self.total_records + self.datadirs_rec_count[i]


    def load_dir_level_ID_batch(self, index):
        ''
        tmp_cnt = 0
        self.list_IDs = []
        self.labels = []
        sdir = len(self.datadirs)
        self.dir_batch_index = index
        self.bstart = index * self.dir_batch_size
        self.benddd = (index * self.dir_batch_size) + self.dir_batch_size
        for i in range(0, sdir):
            if tmp_cnt + self.datadirs_rec_count[i] >= self.bstart:
                for name in os.listdir(self.datadirs[i]):
                    if tmp_cnt >= self.bstart:
                        if tmp_cnt < self.benddd:
                            self.list_IDs.append( os.path.join( self.datadirs[i], name ) ) 
                            self.labels.append( randint(0, self.n_classes) )
                        else:
                            break

                    tmp_cnt = tmp_cnt + 1

            if tmp_cnt < self.benddd:
                tmp_cnt = tmp_cnt + self.datadirs_rec_count[i] 
            else:
                break

        #reset, useful especially when iterator is called manually
        if index == self.__len__():
            self.dir_batch_index = 0

    def init_random_labels(self, list_IDs_len, n_classes):
        'usefull for unsupervised learning.'
        return np.random.choice( n_classes, list_IDs_len )


    def __len__(self):
        'Denotes the number of batches per epoch'
        if not self.is_dir_based_data_batches:
            return int(np.floor(len(self.list_IDs) / self.batch_size))
        else:
            print( "__len__", self.total_records, self.batch_size, int(np.floor(self.total_records / self.batch_size)) )
            return int(np.floor(self.total_records / self.batch_size))

    def __getitem__(self, index, is_return_tuple=True, is_return_only_x=True, is_keep_shuffled_index = False, is_return_last_paths = False ):
        'Generate one batch of data'

        #
        if self.is_dir_based_data_batches:
            if index*self.batch_size >= self.benddd:
                self.load_dir_level_ID_batch(self.dir_batch_index + 1)    
                self.indexes = np.arange(len(self.list_IDs))

                if self.shuffle == True:
                    if is_return_last_paths:
                        self.indexes = self.shuffle_cache[index]
                    else:
                        np.random.shuffle(self.indexes)

                if is_keep_shuffled_index:
                    if self.shuffle_cache is None:
                        self.shuffle_cache = {}

                    self.shuffle_cache[index] = self.indexes


        # Generate indexes of the batch
        indexes = self.indexes[ np.mod(index*self.batch_size, self.dir_batch_size) : np.mod((index+1)*self.batch_size, self.dir_batch_size) ]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        #
        if is_return_last_paths:
            return list_IDs_temp

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        print("__getitem__")
        print(X.shape)
        if self.is_normalize_image_datatype:

            #reshapre
            X = X.reshape((X.shape[0], -1))

            #normalize
            X = np.divide(X, 255.)

        print(X.shape)
        print(y.shape)

        #if its reached the end of it then call on_epoch_end it's necessary in case of explicit loop calls in the code
        if (index+1) == self.__len__():
            self.on_epoch_end( is_xplicit_call = True )

        if is_return_tuple:
            if is_return_only_x:
                return X, X
            else:
                return X, y
        else:
            return X

    def on_epoch_end(self, is_xplicit_call=False):
        'Updates indexes after each epoch'
        if not is_xplicit_call:
            print( "on_epoch_end called..." )
        else:
            print( "on_epoch_end called explicitly..." )

        if not self.is_dir_based_data_batches:
            self.indexes = np.arange(len(self.list_IDs))

            if self.shuffle == True:
                np.random.shuffle(self.indexes)
        else:
            self.load_dir_level_ID_batch(0)
            self.indexes = np.arange(len(self.list_IDs))

            # #TODO better to do dir array shuffle in case dir based batches
            # if self.shuffle == True:
            #     np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.n_channels > 1:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
        else:
            X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if self.datatype == '.npy':
                X[i,] = np.load( ID )   #absolute or relative path to data file
            elif self.datatype == 'imgs_to_gray':
                #print( "ID", ID )
                X[i,] = cv2.cvtColor(cv2.imread( ID ), cv2.COLOR_BGR2GRAY)    #absolute or relative path to record file
            elif self.datatype == 'json':
                X[i,] = cv2.cvtColor(cv2.imread( ID ), cv2.COLOR_BGR2GRAY)    #TODO read as json array

            # Store class
            y[i] = self.labels[i]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes) if self.is_label_to_categorical == True else y
