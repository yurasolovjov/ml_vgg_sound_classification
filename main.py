from __future__ import print_function

import numpy as np
import datetime as dt
from scipy.io import wavfile
import six
import tensorflow as tf
import sys
import argparse
import glob
import pickle
import termcolor
import os
import keras
from sklearn.model_selection import StratifiedKFold

import vggish_params
import vggish_postprocess
import vggish_slim
import vggish_input

from model_dnn import VggDNN
from batcher import DataGenerator,Batcher

import vad
import average_precision


flags = tf.app.flags

flags.DEFINE_string(
    'wav_file', None,
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS

splitter = ","
import sys, os


def print_result(file,list):

    with open(file,"w") as f:
        for l in list:
            f.write("{}\n".format(l))


def getExecPath():
    try:
        sFile = os.path.abspath(sys.modules['__main__'].__file__)
    except:
        sFile = sys.executable
    return os.path.dirname(sFile)

def find_type_class(file_path, pattern, pos = 0,token = splitter):
    with open(file_path,"r") as file:
        for data in file.readlines():
            value = data.replace("\n","").split(token)
            if(value[pos] == str(pattern)):
                return value[0],value[1]
        file.close()

    return -1,-1

def find_id_class(file_type_class,file_id_class,pattern):

    type = find_type_class(file_type_class,pattern)

    if(type[1] != -1):
        return find_type_class(file_id_class,type[1], pos = 1);

    return -1,-1

def getCatalogName(file):
    tail, head = os.path.split(file)

    if tail :
        t, h = os.path.split(tail)

        try:
            return int(h)
        except:
            return -1

    return -1

def makeWaveFilesList(catalogs):

    listWavFiles = list()

    for catalog in catalogs:
        catalog = catalog + "//*.wav"
        wavs = glob.glob(catalog);
        for wav in wavs:
            listWavFiles.append(wav);

    return listWavFiles, len(listWavFiles)

def mergeSerializeData(list1, list2):

    merge_list = list()

    for el1 in list1:
        et_file = el1['file_name']
        for el2 in list2:
            file = el2['file_name']

            if file == et_file:
                id = el1['label_id']
                feat = el2['feat']
                data = {"file_name":file, "label_id":int(id), "feat": feat }
                merge_list.append(data)

    return merge_list

def loadSerialiezedFile(file, version = 1):
    if version == 1:
        with open(file,"rb") as f:
            serializeData = pickle.load(f)

            if os.path.exists(file + ".list"):
                with open(file + ".list","rb") as f2:
                    classData = pickle.load(f2)
                    serializeData = mergeSerializeData(classData,serializeData)

                    if not os.path.exists(file + ".merge"):
                        with open(file + ".merge","wb") as f3:
                            pickle.dump(serializeData,f3)

            return serializeData
    else:
        with open(file, "rb") as f:
            serializeData = pickle.load(f)

            return serializeData

def loadSerializedListFiles(file):

    if os.path.isfile(file):
        catalog,head = os.path.split(file)

        if not catalog:
            raise  Exception("Can not load serialized files")
    elif os.path.isdir(file):
        catalog = file

    catalog = catalog + "//*.features"
    features = glob.glob(catalog);

    serializeData = list()

    for feature in features:
        with open(feature,"rb") as f:
            data = pickle.load(f)
            tail,head = os.path.split(feature)
            type_ser_file = head.split(".")

            try:
                type = int(type_ser_file[-2])
                serializeData.append(data)
            except:
                for element in data:
                    serializeData.append(element)


    return serializeData

def main(_):

    parser = argparse.ArgumentParser(description="Options");
    parser.add_argument("-i","--input", help="Input of catalog", default="./");
    parser.add_argument("-s","--save_serialized_file", default="../serialized_file.pickle",help="Save serialiazed file");
    parser.add_argument("-l","--load_serialized_files", help="Load serialiazed files from catalog");
    parser.add_argument("-e","--etalon_class_file", help="Load etalon file of class");
    parser.add_argument("-t","--etalon_class_id",  help="Load etalon file with content of class id");
    parser.add_argument("-m","--merge",  help="Number seconds for merge", default=1);
    parser.add_argument("-v","--vad",  help="use VAD optimization", default=True);
    parser.add_argument("--save_list_files",  help="Serialize list of files", default=False);
    parser.add_argument("--layer",  help="Layer of serialized data", default='embedding');
    parser.add_argument("--load_test_data",  help="Load test data from catalog");
    parser.add_argument("--models",  help="Catalog for save/load checkpoint of models");
    parser.add_argument("--folds",  help="Number of folds", default=int(3));
    parser.add_argument("--lr",  help="learning rate", default=float(0.0004));
    parser.add_argument("--lr_lim",  help="limit of learning rate", default=float(0.0001));
    parser.add_argument("--factor",  help="new_lr = factor * lr", default=float(0.8));
    parser.add_argument("--tensorboard",  help="new_lr = factor * lr", default="tensorboard");
    parser.add_argument("--batch_size",  help="Batch size", default=int(32));
    parser.add_argument("--scheduler_mode",  help="Scheduler of learning rate", default=None);
    parser.add_argument("--optimizer",  help="Optimizer of learning rate", default='SGD');

    args = parser.parse_args();

    checkPath          = os.path.normpath(str(args.models));
    loadSerializeFiles = args.load_serialized_files;
    loadTestData = args.load_test_data;
    etalon_class_file = args.etalon_class_file;
    etalon_class_id   = args.etalon_class_id;
    merge_sec         = args.merge
    vad_optimization  = bool(args.vad)
    save_list_files = bool(args.save_list_files)
    saveSerilizationFile = args.save_serialized_file;
    inputCatalog = args.input;
    layer = args.layer
    folds = int(args.folds)
    lr = float(args.lr)
    lr_lim = float(args.lr_lim)
    factor = float(args.factor)
    tboard = str(args.tensorboard)
    batch_size = int(args.batch_size)
    scheduler_mode = args.scheduler_mode
    optimizer = str(args.optimizer)

    if checkPath == "None":
        checkPath = getExecPath()
        t,h = os.path.split(checkPath)
        dtn = str(dt.datetime.now()).split(" ")
        dtn = str(dtn[1]).split(".")
        dtn = str(dtn[0]).replace(":","_")
        checkPath = t + '//nnmodels_' +dtn

    checkPath = os.path.normpath(checkPath)
    t,h = os.path.split(checkPath)
    tboard = t + '//' + tboard + "//" + h

    if not os.path.exists(checkPath):
        os.makedirs(checkPath)

    checkPath = checkPath + '//'+str(layer)+'_lr_'+ str(lr) + '_factor_'+str(factor)+'_folds_'+str(folds)
    checkPath = os.path.normpath(checkPath)
    _,h = os.path.split(checkPath)
    tboard = os.path.normpath(tboard + '//' + h)


    if save_list_files:

        serializeData = loadSerializedListFiles(loadTestData)

        with open(os.path.normpath("D:\\repo\\ML\\test_post_competition.csv")) as f:

            et_files = list()

            for line in f.readlines():
                block = line.split(",")
                et_files.append([block[0],block[1]])

        new_merge_data = list()
        count = len(serializeData)
        pos = 0
        for sdata in serializeData:
            fname = sdata['file_name']
            features = sdata['features']

            for label_et in et_files:


                if(str(fname) == str(label_et[0]) and str(label_et[1]) != "None" ):
                    label_id,_ =  find_type_class(etalon_class_id,label_et[1], pos=1);
                    features_data = {"file_name":fname, "label_id":int(label_id), "features": features}
                    new_merge_data.append(features_data)
                    break

            pos = pos + 1

            status_string = str(pos)  + "/" + str(count)
            print(termcolor.colored(status_string,"green"))

        t,h = os.path.split(loadTestData)
        h = str(h).split(".")
        save_list = os.path.normpath(str( t + h[0] +"_label.features"))
        with open(save_list,"wb") as f2:
            pickle.dump(new_merge_data,f2)
            print(termcolor.colored("save: " + save_list,"green"))









        return

    if loadSerializeFiles:

        serializeData = loadSerializedListFiles(loadSerializeFiles)

        if not serializeData:
            raise Exception("Can not unpack serialized data");

        skf = StratifiedKFold(n_splits=folds)
        y = np.asarray([labels['label_id'] for labels in serializeData])

        modelsPath,_ = os.path.split(checkPath)
        modelsPath = modelsPath + "//*.hdf5"
        models = glob.glob(modelsPath);

        if len(models) > 0:
            for i in models:
                print(termcolor.colored(str(i),"green"))
        else:
            print(termcolor.colored(str("Not find models in catalog:" + checkPath),"red"))

        if len(models) == 0:
            i = int(0)
            for train_index, valid_index in skf.split(serializeData,y):

                train_data = DataGenerator(serializeData,train_index, count_class=41, batch_size=batch_size, koeff_merge=int(merge_sec), layer=layer)
                valid_data = DataGenerator(serializeData,valid_index, count_class=41, batch_size=batch_size, koeff_merge=int(merge_sec), layer=layer)

                shape = train_data.get_shape()

                dnn = VggDNN(input_shape=shape, lr=lr, optimizer=optimizer)

                postfix = '_'+str(i)+'.hdf5'
                dnn_model_path = os.path.normpath(checkPath + postfix)
                tboard = os.path.normpath(tboard + postfix)

                dnn.train( train_data,valid_data, checkPath=dnn_model_path, batch_size=batch_size, factor=float(factor), tensorboardPath=tboard, lim_lr=lr_lim, scheduler_mode = scheduler_mode, iteration=train_data.__len__())
                i = i + 1
                # del dnn

            models = glob.glob(modelsPath);

        # batch_data.deleted_garbage()

        actual = list()
        predicts = list()

        pos = int(0)

        # status_string = "Number of models: " + str(len(models))
        # print(termcolor.colored(status_string,"green"))

        dnn_models = list()
        for m in models:
            dnn_models.append(VggDNN(path=m))


        # resuls_predict_string = list()
        #
        # resuls_predict_string.append(["fname,label"])
        # #
        # # for soundfiles,labels,names in batch_data.get_soundfiles():
        # #         count = batch_data.get_countfiles()
        # #
        # #         for sound,label,name in zip(soundfiles,labels,names):
        # #
        # #             predict_merge = np.empty(shape=(0,41));
        # #
        # #             for model in dnn_models:
        # #
        # #                 # sound_ex = np.expand_dims(sound, axis=2)
        # #                 # predict = model.predict_on_batch(np.asarray(sound_ex))
        # #                 predict = model.predict_on_batch(np.asarray(sound))
        # #                 # mean_predict = np.mean(predict, axis=0)
        # #                 # mean_predict = mean_predict.reshape(np.shape(predict)[1],1)
        # #                 predict_merge = np.concatenate((predict_merge,predict), axis=0)
        # #
        # #             mean = np.mean(np.asarray(predict_merge),axis=0)
        # #             amax = np.argsort(mean, axis=0)
        # #             amax = amax[::-1]
        # #
        # #             predict_string = name
        # #
        # #             if etalon_class_id:
        # #                 for r in amax[0:3]:
        # #                     predict_string = predict_string + str(" ") + str(find_type_class(etalon_class_id,r)[1])
        # #
        # #                 predict_string = predict_string + " origin: "+ str(find_type_class(etalon_class_id,np.argmax(label)))
        # #
        # #             resuls_predict_string.append(predict_string)
        # #
        # #             actual.append([np.argmax(label)])
        # #             predicts.append(list(amax))
        # #
        # #             pos = pos + 1
        # #             status_string = "Calculate predict: " + str(pos) + "/" + str(count)
        # #             print(termcolor.colored(status_string,"green"))
        # #             # print(termcolor.colored(str(mean),"green"))
        #
        # met = average_precision.mapk(actual,predicts,k=3)
        # result_string = "Predict: "+str(met)
        #
        # print(termcolor.colored(result_string,"green"))
        # # print(termcolor.colored(resuls_predict_string,"green"))
        # rp = getExecPath()
        # t,h = os.path.split(rp)
        # t = t + "//result.log"
        # print_result(t,resuls_predict_string)


        pos = int(0)

        result_predict_string = list()
        result_predict_string.append(str("fname,label"))

        actual = list()
        predicts = list()

        if loadTestData and os.path.isdir(loadTestData):
            test_data = loadSerializedListFiles(loadTestData)
            batch_data = Batcher(test_data, layer = layer, koeff_merge=int(merge_sec), shuffle= True, n_splits= 2)

            for soundfiles,labels,names in batch_data.get_soundfiles():
                count = batch_data.get_countfiles()

                for sound,label,name in zip(soundfiles,labels,names):

                    predict_merge = np.empty(shape=(0,41));

                    for model in dnn_models:
                        predict = model.predict_on_batch(np.asarray(sound))
                        predict_merge = np.concatenate((predict_merge,predict), axis=0)

                    mean = np.mean(np.asarray(predict_merge),axis=0)
                    amax = np.argsort(mean, axis=0)
                    amax = amax[::-1]

                    predict_string = name + str(",")

                    if etalon_class_id:
                        for r in amax[0:3]:
                            predict_string = predict_string + str(" ") + str(find_type_class(etalon_class_id,r)[1])

                    result_predict_string.append(predict_string)

                    actual.append([np.argmax(label)])
                    predicts.append(list(amax))

                    pos = pos + 1
                    status_string = "Calculate predict: " + str(pos) + "/" + str(count)
                    print(termcolor.colored(status_string,"green"))

            met = average_precision.mapk(actual,predicts,k=3)
            result_string = "Predict test data: "+str(met)

            print(termcolor.colored(result_string,"green"))

            rp = getExecPath()
            t,h = os.path.split(rp)
            t = t + "//test_result.log"
            print_result(t,result_predict_string)

        return

    Catalogs = inputCatalog + "/*/"
    Catalogs = glob.glob(Catalogs);

    Catalogs = Catalogs + [inputCatalog]

    listWavFiles,countFiles = makeWaveFilesList(Catalogs)


    if not listWavFiles:
        print(inputCatalog+": this catalog has not wav files")
        return
    else:


        processedFiles = 0

        for wav_file in listWavFiles:
            processedFiles = processedFiles + 1
            error_string = "Input file : " + wav_file + " - "

            if( os.path.getsize(wav_file) == 0):
                print(termcolor.colored(error_string,"red"))
                continue

            sample_rate, wav_data = wavfile.read(wav_file)

            if vad_optimization == True:
                wav_data = vad.apply_vad(wav_data,sample_rate)

            assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
            samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]

            if(np.shape(samples)[0] < sample_rate):
                append_size = sample_rate - np.shape(samples)[0]
                samples = np.append(samples,np.full(append_size,float(0)))

            examples_batch = vggish_input.waveform_to_examples(samples, sample_rate)

            # Prepare a postprocessor to munge the model embeddings.
            pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

            try:
                with tf.Graph().as_default(), tf.Session() as sess:
                    # Define the model in inference mode, load the checkpoint, and
                    # locate input and output tensors.
                    vggish_slim.define_vggish_slim(training=False)
                    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
                    features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
                    embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
                    flatten_result = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_FL_NAME)

                    # Run inference and postprocessing.
                    [embedding_batch, flatten_batch] = sess.run([embedding_tensor, flatten_result],
                                                 feed_dict={features_tensor: examples_batch})

                    postprocessed_batch = pproc.postprocess(embedding_batch)

                    wav_file_name = os.path.basename(wav_file)
                    label_id, _ = find_id_class(etalon_class_file,etalon_class_id,wav_file_name)

                    if label_id == -1:
                        label_id = getCatalogName(wav_file)

                    features_batch = {"embedding":embedding_batch, "flatten":flatten_batch, "postprocessing":postprocessed_batch}
                    features_data = {"file_name":wav_file_name, "label_id":int(label_id), "features": features_batch}

                    if saveSerilizationFile :
                        with open(saveSerilizationFile + "."+str(processedFiles)+ ".features" ,"wb") as f:
                            pickle.dump(features_data,f)

                    error_string = error_string + "successful"

                    color = "green"

                    if label_id == -1:
                        color = "yellow"
                        error_string = error_string + ". File is not classification"

                    error_string = error_string + " ("+str(processedFiles)+"/"+str(countFiles)+")";
                    print(termcolor.colored(error_string,color))
            except:
                error_string = error_string + "failed"
                error_string = error_string + " ("+str(processedFiles)+"/"+str(countFiles)+")";
                print(termcolor.colored(error_string,"red"))
                continue


        if saveSerilizationFile :

            tail, head = os.path.split(saveSerilizationFile)

            serilizationFilesList = tail + "//*.features"
            serilizationFilesList = glob.glob(serilizationFilesList);

            features_data_merge = list()

            for sfeature in serilizationFilesList:
                with open(sfeature, "rb") as fd:
                    features_data = pickle.load(fd)
                    features_data_merge.append(features_data)

                os.remove(sfeature)

            with open(saveSerilizationFile + ".features", "wb") as f:
                pickle.dump(features_data_merge,f);

if __name__ == '__main__':

    tf.app.run()
