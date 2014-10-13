#!/usr/bin/env python
import logging
import tables
from collections import OrderedDict
from scikits.audiolab import Sndfile
from scikits.audiolab import available_file_formats
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import init
import config
from ground_truth import GroundTruth
from spectrogram import Spectrogram
from track import Track


def list_audio_files_and_genres(audio_folder, extensions):
    files = OrderedDict()
    logging.info("Ground Truth: listing files...")
    for root, dirnames, filenames in os.walk(audio_folder):
        for f in filenames:
            for x in extensions:
                if f.endswith(x):
                    filename = os.path.join(root, f)
                    genre = os.path.basename(root)
                    files[filename] = genre
                    logging.debug("Ground Truth: %s is %s." % (filename, genre))
    logging.info("Ground Truth: %i audio files found." % len(files))
    return files


def check_audio_file_specs(sndfile, samplerate, encoding, channels):
    if sndfile.samplerate != samplerate:
        raise StandardError("%s\nSample rate of above file doesn't match required sample rate of %i", f, samplerate)
    if sndfile.encoding != encoding:
        raise StandardError("%s\nEncoding of above file doesn't match required encoding %s", f, encoding)
    if sndfile.channels != channels:
        raise StandardError("%s\nNumber of channels of above file doesn't match required number of channels (%i)", f, channels)

if __name__ == '__main__':
    init.init_logger()
    conf = config.get_config()
    audio_folder = os.path.expanduser(conf.get('Input', 'AudioFolder'))
    ground_truth_path = os.path.expanduser(conf.get('Preprocessing', 'GroundTruthPath'))
    features_path = os.path.expanduser(conf.get('Preprocessing', 'RawFeaturesPath'))
    extensions = available_file_formats()
    lengthinseconds = int(conf.get('Tracks', 'LengthInSeconds'))
    samplerate = int(conf.get('Tracks', 'SampleRate'))
    encoding = conf.get('Tracks', 'Encoding')
    channels = int(conf.get('Tracks', 'Channels'))
    windowsize = int(conf.get('Spectrogram', 'WindowSize'))
    stepsize = int(conf.get('Spectrogram', 'StepSize'))
    windowtype = conf.get('Spectrogram', 'WindowType')
    fftres = int(conf.get('Spectrogram', 'FFTResolution'))
    if os.path.isdir(audio_folder):
        # ground truth
        audiofiles = list_audio_files_and_genres(audio_folder, extensions)
        gt = GroundTruth(audiofiles)
        gt.save_to_pickle_file(ground_truth_path)
        logging.info("Ground Truth: saved in %s" % ground_truth_path)

        # feature extraction
        logging.info("Feature Extraction: Calculating %i spectrograms... (this may take a while)" % len(gt.ground_truth))
        h5file = tables.open_file(features_path, mode="w", title="Features")
        table = h5file.create_table("/", 'track', Track, "Track")
        tr = table.row
        i = 0
        for filename, genre in gt.ground_truth.iteritems():
            # Read file
            f = Sndfile(filename, mode='r')

            # Check against specs
            check_audio_file_specs(f, samplerate, encoding, channels)

            # Read
            logging.debug("Reading %s" % filename)
            frames = f.read_frames(lengthinseconds*samplerate)

            # Calculate Spectrogram
            logging.debug("Calculating spectrogram of %s" % filename)
            sg = Spectrogram.from_waveform(frames, windowsize, stepsize, windowtype, fftres)

            # Save in feature file
            tr['idnumber'] = i
            tr['name'] = os.path.basename(filename)
            tr['path'] = filename
            tr['genre'] = genre
            # tr['target'] = [gen == genre for gen in gt.genres]
            tr['target'] = gt.genres.index(genre)
            tr['spectrogram'] = sg.spectrogram
            logging.debug("Saving %s in HDF5 file." % filename)
            tr.append()
            i = i + 1
        h5file.close()
        logging.info("Feature Extraction: spectrograms saved in %s." % features_path)

    else:
        raise StandardError("%s not found!" % audio_folder)
