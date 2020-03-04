#!/usr/bin/env python
import os
import numpy as np
import pickle

import toolkits
import model
import preprocess

# path should be in the strucutre of speakerId/
dataPath = '../../dataset/voxceleb/1/train/wav'

def extractSpeakerId( filePath ):
    filePath = filePath.strip()
    tempData = filePath.split('/')
    return tempData[-1]

def getListOfFiles( filePath ):
    returnList = []
    for (dirpath, dirnames, filenames) in os.walk( filePath ):
        if len( filenames ) > 0:
            for filename in filenames:
                returnList.append( os.path.join( dirpath, filename ) )
    return returnList

def main():

    # gpu configuration
    toolkits.initialize_GPU( args )

    #   get speaker id from folder name
    totalList = [os.path.join( dataPath, file) for file in os.listdir( dataPath )]
    uniqueList = np.unique(totalList)
    speakerList = [extractSpeakerId(u) for u in uniqueList]

    #   get audio file for each speaker
    speakerAudioDict = {}
    for speaker in speakerList:

        #   root path
        rootPath = os.path.join( dataPath, speaker )

        #   get list of files
        fileList = getListOfFiles( rootPath )

        #   add to dict
        speakerAudioDict[speaker] = fileList
    
    #   get embedding for each audio of speaker
    speakerToFeatureDict = {}
    for speaker in speakerList:
    
        # construct the data generator.
        params = {  'dim': (257, None, 1),
                    'nfft': 512,
                    'min_slice': 720,
                    'win_length': 400,
                    'hop_length': 160,
                    'n_classes': 5994,
                    'sampling_rate': 16000,
                    'normalize': True,
                }

        network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                    num_class=params['n_classes'],
                                                    mode='eval', args=args)

        # ==> load pre-trained model ???
        if args.resume:
            # ==> get real_model from arguments input,
            # load the model if the imag_model == real_model.
            if os.path.isfile(args.resume):
                network_eval.load_weights(os.path.join(args.resume), by_name=True)
                print('==> successfully loading model {}.'.format(args.resume))
            else:
                raise IOError(
                    "==> no checkpoint found at '{}'".format(args.resume))
        else:
            raise IOError('==> please type in the model to load')

        feats = []
        for ID in speakerAudioDict[speaker]:
            specs = preprocess.load_data(ID, split=False, win_length=params['win_length'], sr=params['sampling_rate'],
                                        hop_length=params['hop_length'], n_fft=params['nfft'],
                                        min_slice=params['min_slice'])
            specs = np.expand_dims(np.expand_dims(specs[0], 0), -1)

            v = network_eval.predict(specs)
            feats += [v]
        speakerToFeatureDict[speaker] = feats

    #   save to file
    with open('speaker_data.pickle', 'wb') as handle:
        pickle.dump( speakerToFeatureDict, handle, protocol=pickle.HIGHEST_PROTOCOL )

if __name__ == "__main__":
    
    # ===========================================
    #        Parse the argument
    # ===========================================
    import argparse
    parser = argparse.ArgumentParser()
    # set up training configuration.
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--resume', default=r'pretrained/weights.h5', type=str)
    # set up network configuration.
    parser.add_argument('--net', default='resnet34s',
                        choices=['resnet34s', 'resnet34l'], type=str)
    parser.add_argument('--ghost_cluster', default=2, type=int)
    parser.add_argument('--vlad_cluster', default=8, type=int)
    parser.add_argument('--bottleneck_dim', default=512, type=int)
    parser.add_argument('--aggregation_mode', default='gvlad',
                        choices=['avg', 'vlad', 'gvlad'], type=str)
    # set up learning rate, training loss and optimizer.
    parser.add_argument('--loss', default='softmax',
                        choices=['softmax', 'amsoftmax'], type=str)
    parser.add_argument('--test_type', default='normal',
                        choices=['normal', 'hard', 'extend'], type=str)

    global args
    args = parser.parse_args()
    main()
