import logging
import config
import theano

log_levels = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}


def init_logger():
    conf = config.get_config()
    log_level = conf.get('Output', 'LogLevel')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s: %(message)s", level=log_levels[log_level])


def init_theano():
    theano.config.floatX = 'float32'
