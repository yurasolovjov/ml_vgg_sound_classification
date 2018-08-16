import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import keras




class Batcher(object):
    def __init__(self, data, layer = 'embedding', count_class = 41, koeff_merge = 1, normalized = True, shuffle = True, n_splits = 3):
        try:
            self.model_position = 0
            self.sound_position = 0
            self.append_size = count_class
            self.koeff_merge = koeff_merge
            self.train_batch = list()
            self.valid_batch = list()
            self.original_valid_batchs = list()
            self.batch = list()
            self.lengthSoundFiles = list()

            skf = StratifiedKFold(n_splits=n_splits)
            X,y = data, np.asarray([labels['label_id'] for labels in data ])

            i = int(0)
            for train_ind, valid_ind in skf.split(X,y):
                print("split: ",str(i))
                i = i + 1
                train_data = np.asarray(data)[train_ind]
                valid_data = np.asarray(data)[valid_ind]
                train_batch,valid_batch,length = self.create_batch(train_data,valid_data, layer, shuffle)
                self.batch.append([train_batch, valid_batch])
                self.lengthSoundFiles.append(length)
                # del train_data, valid_data

            self.size = len(self.batch);

            # if (normalized):
            #     self.normalize(self.train_batch,0.5)
            #     self.normalize(self.valid_batch,0.5)

        except:
            raise Exception("Batcher is failed")

    def create_batch(self,train_set,valid_set,layer = 'embedding',shuffle = True):

        try:

            train_batch = list()
            valid_batch = list()
            lengthSoundFiles = list()

            for train in train_set:
                chunck,self.shape,_ = self.transform_data_in_batch(train['features'][layer], train['label_id'],self.koeff_merge)

                for sec in chunck:
                    train_batch.append(sec)

            for valid in valid_set:
                chunck,shape,l = self.transform_data_in_batch(valid['features'][layer], valid['label_id'],self.koeff_merge)
                lengthSoundFiles.append([l,valid['file_name']])
                # del valid['features']

                for sec in chunck:
                    valid_batch.append(sec)

            if shuffle :
                self.original_valid_batchs.append( valid_batch.copy() )
                np.random.shuffle(train_batch)
                np.random.shuffle(valid_batch)

            return train_batch, valid_batch, lengthSoundFiles

        except:
            raise Exception("create_batch() is failed ");

    def count_batch(self):
        return self.size

    def get_batch(self):

        if self.model_position + 1 == self.size:
            self.model_position = 0

        train_features = list()
        train_labels = list()
        valid_features = list()
        valid_labels = list()

        batch = list()
        batch = self.batch[self.model_position]
        del self.batch[self.model_position]
        print("deleted usebatch ************************")

        for f,l in batch[0]:
            train_features.append(f)
            train_labels.append(l)

        for f,l in batch[1]:
            valid_features.append(f)
            valid_labels.append(l)

        # self.model_position = self.model_position + 1
        return np.asarray(train_features),np.asarray(train_labels),np.asarray(valid_features),np.asarray(valid_labels),self.getShape()

    def get_original_batch(self):

        train_features = list()
        train_labels = list()
        valid_features = list()
        valid_labels = list()

        for f,l in self.original_train_batch:
            train_features.append(f)
            train_labels.append(l)

        for f,l in self.original_valid_batch:
            valid_features.append(f)
            valid_labels.append(l)

        return np.asarray(train_features),np.asarray(train_labels),np.asarray(valid_features),np.asarray(valid_labels),self.getShape()

    def count_soundfiles(self):
        return len(self.lengthSoundFiles)

    def get_soundfiles(self):

        models = list()
        self.count = int(0)


        for i in range(self.size):

            pos = int(0)

            sounds = list()
            labels = list()
            names = list()

            for sound_length,name in self.lengthSoundFiles[i]:
                begin = pos
                end = pos + sound_length
                arr = self.original_valid_batchs[i][begin:end]
                sound = list()
                label = list()

                for s in arr:
                    sound.append(s[0])
                    label.append(s[1])

                self.count = self.count + 1
                sounds.append(sound)
                labels.append(label)
                names.append(name)

                pos = pos + sound_length

            models.append([sounds,labels,names])

        return models
        # return np.asarray(models)

    def get_countfiles(self):
        return int(self.count)
    def transform_data_in_batch(self, data, label_id, koeff_merge = 1 ):

        try:

            batch = list()

            likelihood = np.full(self.append_size,float(0))

            if label_id > -1 :
                likelihood[label_id] = 1.0

            count_merge = 0;
            merge_data = np.empty(shape=(0,));

            countFeatures = len(data)

            #Количество элементов не попавших в группу
            balanceFeatures = countFeatures - (countFeatures // koeff_merge) * koeff_merge

            if(balanceFeatures > 0):
                virtualFeaturesCount = koeff_merge - balanceFeatures
            else:
                virtualFeaturesCount = 0


            virtualFeatures = list()

            if virtualFeaturesCount > 0:
                if(countFeatures < koeff_merge):
                    #Виртуальные элементы дополняющие группу
                    for i in range(virtualFeaturesCount):
                        virtualFeatures.append(np.full(data[0].shape, 0))
                elif(countFeatures > koeff_merge):
                    #Виртуальные элементы дополняющие группу
                    virtualFeatures = data[:virtualFeaturesCount]


            for mini_data,i in zip(data,range(countFeatures)):

                merge_data = np.append(merge_data, mini_data)

                if((i == countFeatures - 1) and (len(virtualFeatures) > 0)):
                    for additional in virtualFeatures:
                        merge_data = np.append(merge_data, additional)

                    count_merge = count_merge + len(virtualFeatures) + 1
                else:
                    count_merge = count_merge + 1

                if count_merge == koeff_merge:
                    merge = [np.asarray(merge_data),np.asarray(likelihood)]
                    batch.append(merge)
                    shape = merge_data.shape
                    # self.train_batch.append(merge)
                    # self.shape = merge_data.shape
                    merge_data = np.empty(shape=(0,));
                    count_merge = 0

            # self.lengthSoundFiles.append(len(self.train_batch))
            return batch,shape,len(batch)
        except:
            raise Exception("transform_data_in_batch() is failed ");

    def normalize(self, batch,koeff = 0.5):

        max = 0
        min = 0

        for sublist,_ in batch:
            possible_max = np.max(sublist)
            possible_min = np.min(sublist)

            if( possible_max > max):
                max = possible_max

            if (possible_min < min):
                min = possible_min



        ready_tuple = False
        feature_normalized = list()
        label = list();

        for i,sublist in np.ndenumerate(batch):
            if( i[1] == 0 ):
                feature_normalized = np.asarray(list(map(lambda x: ((x - min)/(max-min)) - koeff, sublist)))

            if( i[1] == 1 ):
                label = sublist
                ready_tuple = True

            if ready_tuple :
                ready_tuple = False
                batch[i[0]] = [feature_normalized,label]

    def getShape(self):
        return self.shape

    def get_size(self):
        l = len(self.train_batch)
        return l

    def deleted_garbage(self):
        del self.batch
        return 0


class DataGenerator(keras.utils.Sequence):

    def __init__(self, set_data, index, count_class = 41,  batch_size = 32, koeff_merge = 1, layer = 'embedding'):
        self.append_size = count_class
        self.batch_size = batch_size
        self.data = np.asarray(set_data)[index]
        self.koeff_merge = koeff_merge
        self.layer = layer
        self.m_shape = self.get_shape()

    def __len__(self):
        return int(np.ceil(len(self.data)/ float(self.batch_size)))

    def __getitem__(self, index):
        begin = index * self.batch_size
        end = (index + 1) * self.batch_size

        data = self.data[ begin : end ]

        batch = self.__data_generator(data=data, layer=self.layer)

        x = list()
        y = list()


        for feature in batch:
            x.append(feature[0])
            y.append(feature[1])

        return np.asarray(x), np.asarray(y)

    # def on_epoch_end(self):
    #     print("")

    def transform_data_in_batch(self, data, label_id, koeff_merge = 1 ):

        try:

            batch = list()

            likelihood = np.full(self.append_size,float(0))

            if label_id > -1 :
                likelihood[label_id] = 1.0

            count_merge = 0;
            merge_data = np.empty(shape=(0,));

            countFeatures = len(data)

            #Количество элементов не попавших в группу
            balanceFeatures = countFeatures - (countFeatures // koeff_merge) * koeff_merge

            if(balanceFeatures > 0):
                virtualFeaturesCount = koeff_merge - balanceFeatures
            else:
                virtualFeaturesCount = 0


            virtualFeatures = list()

            if virtualFeaturesCount > 0:
                if(countFeatures < koeff_merge):
                    #Виртуальные элементы дополняющие группу
                    for i in range(virtualFeaturesCount):
                        virtualFeatures.append(np.full(data[0].shape, 0))
                elif(countFeatures > koeff_merge):
                    #Виртуальные элементы дополняющие группу
                    virtualFeatures = data[:virtualFeaturesCount]


            for mini_data,i in zip(data,range(countFeatures)):

                merge_data = np.append(merge_data, mini_data)

                if((i == countFeatures - 1) and (len(virtualFeatures) > 0)):
                    for additional in virtualFeatures:
                        merge_data = np.append(merge_data, additional)

                    count_merge = count_merge + len(virtualFeatures) + 1
                else:
                    count_merge = count_merge + 1

                if count_merge == koeff_merge:
                    merge = [np.asarray(merge_data),np.asarray(likelihood)]
                    batch.append(merge)
                    shape = merge_data.shape
                    # self.train_batch.append(merge)
                    # self.shape = merge_data.shape
                    merge_data = np.empty(shape=(0,));
                    count_merge = 0

            # self.lengthSoundFiles.append(len(self.train_batch))
            return batch,shape,len(batch)
        except:
            raise Exception("transform_data_in_batch() is failed ");

    def __data_generator(self,data, layer = 'embedding', shuffle = True):

        try:

            train_batch = list()

            for train in data:
                chunck,self.m_shape,_ = self.transform_data_in_batch(train['features'][layer], train['label_id'],self.koeff_merge)

                for sec in chunck:
                    train_batch.append(sec)

            if shuffle :
                np.random.shuffle(train_batch)

            return train_batch

        except:
            raise Exception("create_batch() is failed ");

    def get_shape(self):

        shape = np.asarray(self.data[0]['features'][self.layer]).shape

        return (shape[1] * self.koeff_merge,)
    # def __data_generator(self, list_id_tmp):
