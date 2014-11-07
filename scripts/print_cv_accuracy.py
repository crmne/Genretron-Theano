#!/usr/bin/env python
import numpy
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from parameters import Parameters

if __name__ == '__main__':
    conf = config.get_config()
    output_folder = os.path.expanduser(conf.get('Output', 'OutputFolder'))
    n_folds = int(conf.get('CrossValidation', 'NumberOfFolds'))

    if not os.path.isdir(output_folder):
        raise StandardError("Experiment folder %s does not exists." % output_folder)

    print("Evaluating Cross-validation Scores for experiment %s..." % output_folder)
    parameters_files = ["parameters%i.pkl" % (x + 1) for x in xrange(n_folds)]
    parameters = [Parameters.load_from_pickle_file(os.path.join(output_folder, x)) for x in parameters_files]
    test_accuracies = numpy.array([1.0 - x.test_error for x in parameters])
    print("Accuracy: %0.4f (+/- %0.4f)" % (test_accuracies.mean(), test_accuracies.std() * 2))
