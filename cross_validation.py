import numpy


class KFold(object):
    """KFold CrossValidation with support for validation sets"""
    def __init__(self, n, n_folds=4):
        assert n_folds >= 3
        self.n = n
        self.n_folds = n_folds
        self.runs = []
        for run_n in xrange(self.n_folds):
            run = {'Train': [], 'Valid': [], 'Test': []}
            folds = numpy.split(numpy.arange(self.n), self.n_folds)
            test_idxs = (0 + run_n) % self.n_folds
            valid_idxs = (1 + run_n) % self.n_folds
            run['Test'] = folds[test_idxs]
            run['Valid'] = folds[valid_idxs]
            run['Train'] = numpy.concatenate([x for i, x in enumerate(folds) if i not in {test_idxs, valid_idxs}])
            self.runs.append(run)
