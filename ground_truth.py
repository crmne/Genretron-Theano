import cPickle as pickle


class GroundTruth(object):

    def __init__(self, dict_of_files_and_genres):
        self.ground_truth = dict_of_files_and_genres
        self.genres = sorted(list(set(self.ground_truth.values())))

    @classmethod
    def load_from_pickle_file(cls, filename):
        return cls(pickle.load(open(filename, "rb")))

    def save_to_pickle_file(self, filename):
        pickle.dump(self.ground_truth, open(filename, "wb"))
