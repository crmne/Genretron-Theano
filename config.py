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

config = None


def config_not_found():
    shutil.copyfile(path_to_default_config, path_to_config)
    raise StandardError(
        "%s not found. We created one for you. Please go and edit it."
        % path_to_config)


def default_config_not_changed():
    raise StandardError(
        "Looks like %s has not been changed from default values. Please go and edit it."
        % path_to_config)


def process_config():
    global config
    config = ConfigParser.ConfigParser()
    config.read(path_to_config)
    return config


def print_config():
    config = get_config()
    for section in config.sections():
        print("[%s]: %s" % (section, dict(config.items(section))))


def get_config(refresh=False):
    if config is None or refresh is True:
        if os.path.isfile(path_to_config):
            if filecmp.cmp(path_to_config, path_to_default_config):
                default_config_not_changed()
            else:
                return process_config()
        else:
            config_not_found()
    else:
        return config


def copy_to(filename):
    shutil.copyfile(path_to_config, filename)
