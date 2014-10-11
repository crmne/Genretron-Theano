#!/usr/bin/env python
import logging
import os
import sys
import shutil
import urllib
import tarfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import init
import config
import utils

gtzan_origin = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"

if __name__ == '__main__':
    init.init_logger()
    conf = config.get_config()
    audio_folder = os.path.expanduser(conf.get('Input', 'AudioFolder'))
    if os.path.isdir(audio_folder):
        if utils.query_yes_no("Audio folder %s exists. Do you want to overwrite it?" % audio_folder):
            shutil.rmtree(audio_folder)
            os.makedirs(audio_folder)
    else:
        logging.debug("Audio Folder %s not found. Creating..." % audio_folder)
        os.makedirs(audio_folder)

    gtzan_dest = os.path.join(audio_folder, os.path.basename(gtzan_origin))
    if os.path.isfile(gtzan_dest):
        if utils.query_yes_no("GTZAN dataset already downloaded in %s. Do you want to download it again?" % gtzan_dest):
            os.remove(gtzan_dest)
            logging.info("Downloading GTZAN dataset from %s to %s" % (gtzan_origin, gtzan_dest))
            urllib.urlretrieve(gtzan_origin, gtzan_dest)
    else:
        logging.info("Downloading GTZAN dataset from %s to %s" % (gtzan_origin, gtzan_dest))
        urllib.urlretrieve(gtzan_origin, gtzan_dest)

    logging.info("Extracting audio files to %s" % audio_folder)
    tar = tarfile.open(gtzan_dest, 'r:gz')
    tar.extractall(audio_folder)
    tar.close()

    # flatten dir structure
    genre_dir = os.path.join(audio_folder, 'genres')
    for genre in os.listdir(genre_dir):
        shutil.move(os.path.join(genre_dir, genre), os.path.join(audio_folder, genre))
    os.rmdir(genre_dir)

    logging.info("All done.")
