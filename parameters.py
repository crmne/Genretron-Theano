import cPickle as pickle
import theano


class Parameters(object):

    def __init__(self, params, classifier_name, validation_error, test_error):
        assert isinstance(params[0], theano.tensor.sharedvar.TensorSharedVariable)
        assert isinstance(classifier_name, str)
        self.params = params
        self.classifier_name = classifier_name
        self.validation_error = validation_error
        self.test_error = test_error

    @classmethod
    def load_from_pickle_file(cls, filename):
        params = pickle.load(open(filename, "rb"))
        return cls(
            params['params'],
            params['classifier_name'],
            params['validation_error'],
            params['test_error']
        )

    def save_to_pickle_file(self, filename):
        params = {
            'params': self.params,
            'classifier_name': self.classifier_name,
            'validation_error': self.validation_error,
            'test_error': self.test_error
        }
        pickle.dump(params, open(filename, "wb"))

    def save(self):
        import config
        conf = config.get_config()
        save_parameters = True if conf.get('Output', 'SaveBestModel') == 'Yes' else False
        if save_parameters:
            import os
            output_folder = os.path.expanduser(conf.get('Output', 'OutputFolder'))
            run_n = int(conf.get('CrossValidation', 'RunNumber'))
            output_file = os.path.join(output_folder, 'parameters%i.pkl' % run_n)
            # import logging
            # logging.debug("Parameters and error scores saved in %s" % output_file)
            self.save_to_pickle_file(output_file)
