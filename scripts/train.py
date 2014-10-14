#!/usr/bin/env python
import time
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
from logistic_regression import LogisticRegression


def load_features(features_path):
    logging.info("Loading Features from %s..." % features_path)
    return tables.open_file(features_path, 'r')


def shuffle_in_unison(x, y, random_state):
    rng_state = random_state.get_state()
    random_state.shuffle(x)
    random_state.set_state(rng_state)
    random_state.shuffle(y)


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


def prepare_dataset_from_feature_file(features_path, random_state):
    # This function exists to let the garbage collector remove X and Y from
    # memory
    features = load_features(features_path)
    feature_slice = features.root.track[:]
    features.close()
    X = feature_slice['spectrogram']
    Y = feature_slice['target'].astype(numpy.int32)

    logging.info("Preprocessing data...")
    X_reshaped = numpy.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))
    preprocessing.StandardScaler(copy=False).fit_transform(X_reshaped, Y)

    logging.debug("Shuffling using seed = %i..." % random_state.get_state()[1][0])
    shuffle_in_unison(X_reshaped, Y, random_state)

    logging.debug("Splitting...")
    (train_x, valid_x, test_x), (train_y, valid_y, test_y) = split_dataset(
        X_reshaped, Y, train_valid_test_ratios)

    logging.debug("Loading into Theano shared variables...")
    borrow = False
    train_set_x, train_set_y = shared_dataset(train_x, train_y, borrow=borrow)
    valid_set_x, valid_set_y = shared_dataset(valid_x, valid_y, borrow=borrow)
    test_set_x, test_set_y = shared_dataset(test_x, test_y, borrow=borrow)
    return [(train_set_x, train_set_y),
            (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

if __name__ == '__main__':
    init.init_logger()
    init.init_theano()
    conf = config.get_config()
    train_valid_test_ratios = [int(x) for x in conf.get('Preprocessing', 'TrainValidTestPercentages').split(' ')]
    batch_size = int(conf.get('Model', 'BatchSize'))
    learning_rate = float(conf.get('Model', 'LearningRate'))
    n_epochs = int(conf.get('Model', 'NumberOfEpochs'))
    n_genres = int(conf.get('Tracks', 'NumberOfGenres'))
    patience = int(conf.get('EarlyStopping', 'Patience'))
    patience_increase = int(conf.get('EarlyStopping', 'PatienceIncrease'))
    improvement_threshold = float(conf.get('EarlyStopping', 'ImprovementThreshold'))
    seed = None if conf.get('Model', 'Seed') == 'None' else int(conf.get('Model', 'Seed'))
    features_path = os.path.expanduser(conf.get('Preprocessing', 'RawFeaturesPath'))

    random_state = RandomState(seed)

    datasets = prepare_dataset_from_feature_file(features_path, random_state)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    logging.info("Building the model...")

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    classifier = LogisticRegression(input=x, n_in=train_set_x.get_value(borrow=True).shape[1], n_out=n_genres)

    cost = classifier.negative_log_likelihood(y)

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    logging.info('Training the model...')
    # go through this many minibatche before checking the network on the validation set; in this case we check every epoch
    validation_frequency = min(n_train_batches, patience / 2)

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                logging.info('epoch %i, minibatch %i/%i, validation error %f %%' %
                    (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    logging.info('     epoch %i, minibatch %i/%i, test error of best model %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    logging.info('Optimization complete with best validation score of %f %%, with test performance %f %%' %
          (best_validation_loss * 100., test_score * 100.))
    logging.info('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    logging.info('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % (end_time - start_time))
    # TODO: save parameters and logs in experiment folder
