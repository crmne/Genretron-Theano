import logging
import shutil
import filecmp
import os.path
import ConfigParser

path_to_default_config = os.path.join(
    os.path.dirname(__file__),
    'config.ini.default'
)
path_to_config = os.path.join(
    os.path.dirname(__file__),
    'config.ini'
)


def config_not_found():
    shutil.copyfile(path_to_default_config, path_to_config)
    logging.error(
        "%s not found. We created one for you. Please go and edit it."
        % path_to_config)
    return 1


def default_config_not_changed():
    logging.error(
        "Looks like %s has not been changed from default values."
        % path_to_config)
    return 1


def process_config():
    config = ConfigParser.ConfigParser()
    config.read(path_to_config)
    print(config)

if __name__ == '__main__':
    if os.path.isfile(path_to_config):
        if filecmp.cmp(path_to_config, path_to_default_config):
            default_config_not_changed()
        else:
            process_config()
    else:
        config_not_found()
