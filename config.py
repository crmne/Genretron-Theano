import argparse
import shutil
import filecmp
import os.path
import ConfigParser

config = None


def get_config():
    global config
    if config is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", required=False)
        config = Config(parser.parse_args().config)
    return config.config


def copy_to(filename):
    global config
    if config is not None:
        config.copy_to(filename)


class Config(object):
    path_to_default_config = os.path.join(os.path.dirname(__file__), 'config.ini.default')

    def __init__(self, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), 'config.ini')
        else:
            path = os.path.abspath(path)
        if os.path.isfile(path):
            if filecmp.cmp(path, Config.path_to_default_config):
                raise StandardError("Looks like %s has not been changed from default values. Please go and edit it." % path)
        else:
            shutil.copyfile(Config.path_to_default_config, path)
            raise StandardError("%s not found. We created it for you. Please go and edit it." % path)

        print("Using config %s" % path)
        self.path = path

        self.config = ConfigParser.ConfigParser()
        self.config.read(path)

    def copy_to(self, filename):
        shutil.copyfile(self.path, filename)
