#!/usr/bin/env python
import numpy
from numpy.random import RandomState
import theano
import theano.tensor as T
import logging
import tables
from sklearn import preprocessing
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import init
import config
import utils
from logistic_regression import LogisticRegression
from mlp import MLP
from classifier import Classifier
from cross_validation import KFold


def load_features(features_path):
    logging.info("Loading Features from %s..." % features_path)
    return tables.open_file(features_path, 'r')


def shared_dataset(data_x, data_y, borrow):
    # from theano's logistic regression example
    shared_x = theano.shared(name='shared_x', value=data_x, borrow=borrow)
    shared_y = theano.shared(name='shared_y', value=data_y, borrow=borrow)
    return shared_x, shared_y


def preprocess(X, Y):
    logging.debug("Reshaping...")
    X_reshaped = numpy.reshape(X, (X.shape[0] * X.shape[1], X.shape[2]))
    Y_reshaped = numpy.repeat(Y, X.shape[1], axis=0)
    logging.debug("Scaling...")
    scaler = preprocessing.StandardScaler(copy=False)
    scaler.fit(X_reshaped)
    scaler.transform(X_reshaped)

    return X_reshaped, Y_reshaped


def load_into_theano(train_x, train_y, valid_x, valid_y, test_x, test_y):
    logging.debug("Loading into Theano shared variables...")
    borrow = True
    train_set_x, train_set_y = shared_dataset(train_x, train_y, borrow=borrow)
    valid_set_x, valid_set_y = shared_dataset(valid_x, valid_y, borrow=borrow)
    test_set_x, test_set_y = shared_dataset(test_x, test_y, borrow=borrow)
    return [(train_set_x, train_set_y),
            (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]


def save_preprocessed(_X, _Y, preprocessed_path):
    h5file = tables.open_file(preprocessed_path, mode='w', title='Preprocessed')

    class Data(tables.IsDescription):
        X = tables.Float32Col(shape=_X.shape[1])
        Y = tables.Int32Col(shape=())

    table = h5file.create_table('/', 'data', Data, "Data")
    tr = table.row
    for i in xrange(_X.shape[0]):
        tr['X'] = _X[i]
        tr['Y'] = _Y[i]
        tr.append()

    h5file.close()


def load_dataset_and_preprocess(preprocessed_path, features_path, rng):
    X, Y = None, None
    if os.path.isfile(preprocessed_path):
        features = load_features(preprocessed_path)
        feature_slice = features.root.data[:]
        features.close()

        X = feature_slice['X']
        Y = feature_slice['Y']
    else:
        features = load_features(features_path)
        feature_slice = features.root.track[:]
        features.close()

        logging.info("Preprocessing data...")
        X, Y = preprocess(
            feature_slice['spectrogram'],
            feature_slice['target'].astype(numpy.int32)
        )
        logging.info("Saving preprocessed data...")
        save_preprocessed(X, Y, preprocessed_path)

    n_tracks = 1000
    nw_per_track = 1247  # TODO: make those two numbers dynamic
    tracks = numpy.arange(n_tracks)
    logging.debug("Shuffling tracks...")
    rng.shuffle(tracks)

    logging.debug("Creating folds...")
    n_folds = int(conf.get('CrossValidation', 'NumberOfFolds'))
    run_n = int(conf.get('CrossValidation', 'RunNumber'))
    logging.info("Cross-Validation run number %i/%i" % (run_n, n_folds))
    kf = KFold(tracks, n_folds=n_folds)
    run = kf.runs[run_n - 1]
    logging.debug("Run description: %s" % run)

    def track_ns_to_idxs(n_windows_per_track, track_ns):
        return numpy.array(
            [numpy.arange(
                x * n_windows_per_track,
                (x * n_windows_per_track) + n_windows_per_track
            ) for x in track_ns]).flatten()

    logging.debug("Assigning training data...")
    train_x = numpy.take(X, track_ns_to_idxs(nw_per_track, run['Train']), axis=0)
    logging.debug("Assigning training targets...")
    train_y = numpy.take(Y, track_ns_to_idxs(nw_per_track, run['Train']), axis=0)
    logging.debug("Assigning validation data...")
    valid_x = numpy.take(X, track_ns_to_idxs(nw_per_track, run['Valid']), axis=0)
    logging.debug("Assigning validation targets...")
    valid_y = numpy.take(Y, track_ns_to_idxs(nw_per_track, run['Valid']), axis=0)
    logging.debug("Assigning test data...")
    test_x = numpy.take(X, track_ns_to_idxs(nw_per_track, run['Test']), axis=0)
    logging.debug("Assigning test targets...")
    test_y = numpy.take(Y, track_ns_to_idxs(nw_per_track, run['Test']), axis=0)
    return load_into_theano(train_x, train_y, valid_x, valid_y, test_x, test_y)

activations = {
    'Sigmoid': T.nnet.sigmoid,
    'ReLU': lambda x: T.maximum(0, x),
    'HT': T.tanh,
    'Softplus': T.nnet.softplus,
    'Softmax': T.nnet.softmax
}

if __name__ == '__main__':
    init.init_output()
    init.init_logger()
    init.init_theano()
    conf = config.get_config()
    model = conf.get('Model', 'Model')
    batch_size = int(conf.get('Model', 'BatchSize'))
    learning_rate = float(conf.get('Model', 'LearningRate'))
    n_epochs = int(conf.get('Model', 'NumberOfEpochs'))
    audio_folder = os.path.expanduser(conf.get('Input', 'AudioFolder'))
    n_genres = len(utils.list_subdirs(audio_folder))
    patience = int(conf.get('EarlyStopping', 'Patience'))
    patience_increase = int(conf.get('EarlyStopping', 'PatienceIncrease'))
    improvement_threshold = float(conf.get('EarlyStopping', 'ImprovementThreshold'))
    l1_reg = float(conf.get('MultiLayerPerceptron', 'L1RegularizationWeight'))
    l2_reg = float(conf.get('MultiLayerPerceptron', 'L2RegularizationWeight'))
    n_hidden = int(conf.get('MultiLayerPerceptron', 'NumberOfNeuronsPerHiddenLayer'))  # FIXME: only one layer is supported now
    activation = conf.get('MultiLayerPerceptron', 'Activation')
    seed = None if conf.get('Model', 'Seed') == 'None' else int(conf.get('Model', 'Seed'))
    save_best_model = True if conf.get('Output', 'SaveBestModel') == 'Yes' else False
    output_folder = os.path.expanduser(conf.get('Output', 'OutputFolder'))
    features_path = os.path.expanduser(conf.get('Preprocessing', 'RawFeaturesPath'))
    preprocessed_path = os.path.expanduser(conf.get('Preprocessing', 'PreprocessedFeaturesPath'))
    run_n = int(conf.get('CrossValidation', 'RunNumber'))

    logging.info("Output folder: %s" % output_folder)

    config.copy_to(os.path.join(output_folder, 'config%i.ini' % run_n))

    # Preprocessing
    rng = RandomState(seed)

    datasets = load_dataset_and_preprocess(preprocessed_path, features_path, rng)

    # Training
    x = T.matrix('x')
    n_in = datasets[0][0].get_value(borrow=True).shape[1]

    actual_classifier = LogisticRegression(
        input=x,
        n_in=n_in,
        n_out=n_genres
    ) if model == 'LogisticRegression' else MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        n_hidden=n_hidden,
        activation=activations[activation],
        n_out=n_genres,
        l1_reg=l1_reg,
        l2_reg=l2_reg
    )
    classifier = Classifier(actual_classifier, x, batch_size, learning_rate, datasets)
    classifier.train(
        patience,
        patience_increase,
        n_epochs,
        improvement_threshold
    )
