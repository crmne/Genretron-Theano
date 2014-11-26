import os
import logging
import theano
# import shutil
import config
# import utils

log_levels = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}


def init_output():
    conf = config.get_config()
    output_folder = os.path.expanduser(conf.get('Output', 'OutputFolder'))
    if not os.path.isdir(output_folder):
        try:
            os.makedirs(output_folder)
        except OSError, e:
            if e.errno == 17:  # folder might have been created after checking for its existence
                pass
            else:
                raise


def init_logger():
    conf = config.get_config()
    log_level = conf.get('Output', 'ConsoleLogLevel')
    output_folder = os.path.expanduser(conf.get('Output', 'OutputFolder'))
    run_n = int(conf.get('CrossValidation', 'RunNumber'))
    log_file = os.path.join(output_folder, 'run%i.log' % run_n)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s: %(message)s",
        level=logging.DEBUG,
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(log_levels[log_level])
    logging.getLogger('').addHandler(console)


def init_theano():
    theano.config.floatX = 'float32'
