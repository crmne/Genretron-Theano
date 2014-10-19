import time
import numpy
import theano
import theano.tensor as T
import logging
from plot import Plot
from parameters import Parameters


class Classifier(object):
    def __init__(self, actual_classifier, x, batch_size, learning_rate, datasets):
        self.batch_size = batch_size

        self.train_set_x, self.train_set_y = datasets[0]
        self.valid_set_x, self.valid_set_y = datasets[1]
        self.test_set_x, self.test_set_y = datasets[2]

        # compute number of minibatches for training, validation and testing
        self.n_train_batches = (
            self.train_set_x.get_value(borrow=True).shape[0] / self.batch_size
        )

        self.n_valid_batches = (
            self.valid_set_x.get_value(borrow=True).shape[0] / self.batch_size
        )
        self.n_test_batches = (
            self.test_set_x.get_value(borrow=True).shape[0] / self.batch_size
        )

        logging.debug(
            'Train batches = %i, Valid batches = %i, Test batches = %i' %
            (self.n_train_batches, self.n_valid_batches, self.n_test_batches)
        )

        self.classifier = actual_classifier

        logging.info("Building the model...")

        self.index = T.lscalar()
        self.x = x
        self.y = T.ivector('y')

        self.cost = self.classifier.cost(self.y)

        self.test_model = theano.function(
            inputs=[self.index],
            outputs=self.classifier.errors(self.y),
            givens={
                self.x:
                self.test_set_x[
                    self.index * self.batch_size:(self.index + 1) * self.batch_size
                ],
                self.y:
                self.test_set_y[
                    self.index * self.batch_size:(self.index + 1) * self.batch_size
                ]
            }
        )

        self.validate_model = theano.function(
            inputs=[self.index],
            outputs=self.classifier.errors(self.y),
            givens={
                self.x:
                self.valid_set_x[
                    self.index * self.batch_size:(self.index + 1) * self.batch_size
                ],
                self.y:
                self.valid_set_y[
                    self.index * self.batch_size:(self.index + 1) * self.batch_size
                ]
            }
        )

        # compiling a Theano function `train_model` that returns the cost, but in
        # the same time updates the parameter of the model based on the rules
        # defined in `updates`
        self.train_model = theano.function(
            inputs=[self.index],
            outputs=self.cost,
            updates=self.classifier.updates(self.cost, learning_rate),
            givens={
                self.x:
                self.train_set_x[
                    self.index * self.batch_size:(self.index + 1) * self.batch_size
                ],
                self.y:
                self.train_set_y[
                    self.index * self.batch_size:(self.index + 1) * self.batch_size
                ]
            }
        )

    def train(self, patience, patience_increase, n_epochs, improvement_threshold):
        logging.info('Training the model...')
        plot = Plot('Validation', 'Test')
        # go through this many minibatches before checking the network on the
        # validation set; in this case we check every epoch
        validation_frequency = min(self.n_train_batches, patience / 2)

        best_params = None
        best_validation_loss = numpy.inf
        test_score = 0.
        start_time = time.clock()

        done_looping = False
        epoch = 0
        try:
            while (epoch < n_epochs) and (not done_looping):
                epoch = epoch + 1
                for minibatch_index in xrange(self.n_train_batches):

                    minibatch_avg_cost = self.train_model(minibatch_index)
                    # iteration number
                    iter = (epoch - 1) * self.n_train_batches + minibatch_index

                    if (iter + 1) % validation_frequency == 0:
                        # compute zero-one loss on validation set
                        validation_losses = [self.validate_model(i)
                                             for i in xrange(self.n_valid_batches)]
                        this_validation_loss = numpy.mean(validation_losses)

                        logging.info(
                            'epoch %i, minibatch %i/%i, validation error %f %%' %
                            (
                                epoch,
                                minibatch_index + 1,
                                self.n_train_batches,
                                this_validation_loss * 100.
                            )
                        )

                        plot.append('Validation', this_validation_loss)
                        plot.update_plot()

                        # if we got the best validation score until now
                        if this_validation_loss < best_validation_loss:
                            # improve patience if loss improvement is good enough
                            if this_validation_loss < best_validation_loss *  \
                               improvement_threshold:
                                patience = max(patience, iter * patience_increase)

                            best_validation_loss = this_validation_loss

                            # test it on the test set
                            test_losses = [self.test_model(i)
                                           for i in xrange(self.n_test_batches)]
                            test_score = numpy.mean(test_losses)

                            logging.info(
                                '     epoch %i, minibatch %i/%i test error of best model %f %%' %
                                (
                                    epoch,
                                    minibatch_index + 1,
                                    self.n_train_batches,
                                    test_score * 100.
                                )
                            )

                            plot.append('Test', test_score)
                            plot.update_plot()

                            best_params = Parameters(self.classifier.W.get_value(borrow=True), self.classifier.b.get_value(borrow=True))
                        else:
                            plot.append('Test', numpy.NaN)
                            plot.update_plot()

                    if patience <= iter:
                        done_looping = True
                        break

        finally:
            if best_params is not None:
                best_params.save()
            plot.save_plot()
            end_time = time.clock()
            logging.info(
                'Optimization complete with best validation score of %f %%, with test performance %f %%' %
                (best_validation_loss * 100., test_score * 100.))
            logging.info(
                'The code run for %d epochs, with %f epochs/sec' %
                (epoch, 1. * epoch / (end_time - start_time)))

    def predict(self):
        raise NotImplementedError
