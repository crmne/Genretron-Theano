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
