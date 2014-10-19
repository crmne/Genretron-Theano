import cPickle as pickle
import numpy


class Parameters(object):

    def __init__(self, W, b):
        assert type(W) == numpy.ndarray
        assert type(b) == numpy.ndarray
        self.W = W
        self.b = b

    @classmethod
    def load_from_pickle_file(cls, filename):
        params = pickle.load(open(filename, "rb"))
        return cls(params['W'], params['b'])

    def save_to_pickle_file(self, filename):
        params = {'W': self.W, 'b': self.b}
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
