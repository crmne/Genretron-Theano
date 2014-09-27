#!/usr/bin/env python
import logging
import tables
from scikits.audiolab import sndfile
from scikits.audiolab import available_file_formats
from os import sys, path, walk
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import init
import config
import ground_truth
import spectrogram
from track import Track


def list_audio_files_and_genres(audio_folder, extensions):
    files = {}
    logging.info("Ground Truth: listing files...")
    for root, dirnames, filenames in walk(audio_folder):
        for f in filenames:
            for x in extensions:
                if f.endswith(x):
                    filename = path.join(root, f)
                    genre = path.basename(root)
                    files[filename] = genre
                    logging.debug("Ground Truth: %s is %s." % (filename, genre))
    logging.info("Ground Truth: %i audio files found." % len(files))
    return files


def calc_spectrogram(audio_file, spectrogram_options, tracks_options):
    lengthinseconds = int(tracks_options['lengthinseconds'])
    samplerate = int(tracks_options['samplerate'])
    windowsize = int(spectrogram_options['windowsize'])
    stepsize = int(spectrogram_options['stepsize'])
    windowtype = spectrogram_options['windowtype']
    fftres = int(spectrogram_options['fftresolution'])

    logging.debug("Reading %s" % audio_file)
    f = sndfile(audio_file, mode='read')  # TODO: check if file has good form
    frames = f.read_frames(lengthinseconds*samplerate)
    logging.debug("Calculating spectrogram of %s" % audio_file)
    sg = spectrogram.Spectrogram(frames, windowsize, stepsize, windowtype, fftres)
    return sg.spectrogram


if __name__ == '__main__':
    init.init_logger()
    conf = config.get_config()
    audio_folder = path.expanduser(conf.get('Input', 'AudioFolder'))
    lists_folder = path.expanduser(conf.get('Preprocessing', 'ListsFolder'))
    ground_truth_path = path.join(lists_folder, 'ground_truth.pkl')
    features_folder = path.expanduser(conf.get('Preprocessing', 'FeaturesFolder'))
    features_path = path.join(features_folder, 'features.h5')
    extensions = available_file_formats()
    if path.isdir(audio_folder):
        # ground truth
        audiofiles = list_audio_files_and_genres(audio_folder, extensions)
        gt = ground_truth.GroundTruth(audiofiles)
        gt.save_to_pickle_file(ground_truth_path)
        logging.info("Ground Truth: saved in %s" % ground_truth_path)

        # feature extraction
        logging.info("Feature Extraction: Calculating %i spectrograms... (this may take a while)" % len(gt.ground_truth))
        h5file = tables.open_file(features_path, mode="w", title="Features")
        table = h5file.create_table("/", 'track', Track, "Track")
        tr = table.row
        for filename, genre in gt.ground_truth.iteritems():
            tr['name'] = path.basename(filename)
            tr['path'] = filename
            tr['genre'] = genre
            tr['spectrogram'] = calc_spectrogram(
                filename,
                dict(conf.items('Spectrogram')),
                dict(conf.items('Tracks'))
            )
            logging.debug("Saving %s in HDF5 file." % filename)
            tr.append()
        h5file.close()
        logging.info("Feature Extraction: saved spectrograms in %s." % features_path)

    else:
        raise StandardError("%s not found!" % audio_folder)
