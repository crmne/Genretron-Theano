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
    # if os.path.isdir(output_folder):
    #     if utils.query_yes_no("Output folder %s exists. Do you want to overwrite it?" % output_folder):
    #         shutil.rmtree(output_folder)
    #         os.makedirs(output_folder)
    #     else:
    #         raise StandardError("Output folder %s already exists." % output_folder)
    # else:
    #     os.makedirs(output_folder)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)


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
