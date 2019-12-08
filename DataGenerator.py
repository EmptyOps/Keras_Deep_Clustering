import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, is_init_random_labels=False, datatype='.npy', datadirs=[], is_label_to_categorical=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.datatype = datatype
        self.datadirs = datadirs
        self.is_label_to_categorical = is_label_to_categorical


        if not list_IDs == None:
            self.list_IDs = list_IDs
        else:
            self.__init_IDs__(list_IDs, datadirs)            

        if is_init_random_labels == False:
            self.labels = labels
        else:
            self.labels = self.init_random_labels(list_IDs, n_classes)

        self.on_epoch_end()

    def __init_IDs__(self, datadirs):
        'usefull when number of records is large, to make it easier to even load ids by batch'

        # support multiple dirs to make it easier to train large dataset in different dirs without moving them however to offset the gradient leaning towards 
        # one direction for one dir and for the other on the other direction it is made sure that large enough indexes are loaded together(to keep benefits 
        # of shuffeling) like loaded_indexes = batch_size * 50
        self.datadirs = datadirs
        self.datadirs_rec_count = []

        #count total
        sdir = len(datadirs)
        for i in range(0, sdir):
            self.datadirs_rec_count[i] = len([True for name in os.listdir(self.datadirs[i]) if os.path.isfile(name)])

        #load first dir level batch


    def load_dir_level_ID_batch(self, list_IDs, n_classes):
        ''



    def init_random_labels(self, list_IDs, n_classes):
        'usefull for unsupervised learning. class labels are generated in sequence for reproducibility'


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if len(self.datadirs) <= 1:
            self.indexes = np.arange(len(self.list_IDs))
        else:
            self.indexes = np.arange(len(self.datadirs))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + self.datatype)

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes) if self.is_label_to_categorical == True else y