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


def load_features(features_path):
    logging.info("Loading Features from %s..." % features_path)
    return tables.open_file(features_path, 'r')


def shuffle_in_unison(x, y, rng):
    logging.debug("Seed = %i" % rng.get_state()[1][0])
    rng_state = rng.get_state()
    rng.shuffle(x)
    rng.set_state(rng_state)
    rng.shuffle(y)
    rng.set_state(rng_state)


def split_dataset(data_x, data_y, ratios):
    assert sum(ratios) == 100
    split_points = []
    for ratio in ratios:
        prev = 0
        if len(split_points) != 0:
            prev = split_points[-1]
        split_points.append((len(data_x) * ratio / 100) + prev)
    logging.debug("Split points = %s" % split_points[:-1])
    xs = numpy.split(data_x, split_points[:-1])
    ys = numpy.split(data_y, split_points[:-1])
    return xs, ys


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


def split_and_load_into_theano(X, Y):
    logging.debug("Splitting...")
    (train_x, valid_x, test_x), (train_y, valid_y, test_y) = split_dataset(
        X, Y, train_valid_test_ratios)

    logging.debug("Loading into Theano shared variables...")
    borrow = False
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


def load_dataset_and_preprocess(preprocessed_path, features_path):
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

    logging.debug("Shuffling...")
    shuffle_in_unison(X, Y, rng)

    return split_and_load_into_theano(X, Y)

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
    train_valid_test_ratios = [int(x) for x in conf.get('Preprocessing', 'TrainValidTestPercentages').split(' ')]
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

    logging.info("Output folder: %s" % output_folder)

    config.copy_to(os.path.join(output_folder, 'config.ini'))

    # Preprocessing
    rng = RandomState(seed)

    datasets = load_dataset_and_preprocess(preprocessed_path, features_path)

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
