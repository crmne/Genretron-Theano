import cPickle as pickle
import theano


class Parameters(object):

    def __init__(self, params, classifier_name):
        assert isinstance(params[0], theano.tensor.sharedvar.TensorSharedVariable)
        assert isinstance(classifier_name, str)
        self.params = params
        self.classifier_name = classifier_name

    @classmethod
    def load_from_pickle_file(cls, filename):
        params = pickle.load(open(filename, "rb"))
        return cls(params['params'], params['classifier_name'])

    def save_to_pickle_file(self, filename):
        params = {'params': self.params, 'classifier_name': self.classifier_name}
        pickle.dump(params, open(filename, "wb"))

    def save(self):
        import config
        conf = config.get_config()
        save_parameters = True if conf.get('Output', 'SaveBestModel') == 'Yes' else False
        if save_parameters:
            import os
            output_folder = os.path.expanduser(conf.get('Output', 'OutputFolder'))
            output_file = os.path.join(output_folder, 'parameters.pkl')
            import logging
            logging.info("Parameters saved in %s" % output_file)
            self.save_to_pickle_file(output_file)
