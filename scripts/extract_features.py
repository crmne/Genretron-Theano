#!/usr/bin/env python
import logging
import tables
from scikits.audiolab import sndfile
from os import sys, path, walk
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import config
import ground_truth
import spectrogram
from track import Track


def list_audio_files_and_genres(audio_folder, extensions):
    files = {}
    for root, dirnames, filenames in walk(audio_folder):
        for f in filenames:
            for x in extensions:
                if f.endswith(x):
                    files[path.join(root, f)] = path.basename(root)
    logging.info("%i files found." % len(files))
    return files


def calc_spectrogram(audio_file, spectrogram_options, tracks_options):
    lengthinseconds = int(tracks_options['lengthinseconds'])
    samplerate = int(tracks_options['samplerate'])
    windowsize = int(spectrogram_options['windowsize'])
    stepsize = int(spectrogram_options['stepsize'])
    windowtype = spectrogram_options['windowtype']
    fftres = int(spectrogram_options['fftresolution'])

    f = sndfile(audio_file, mode='read')  # TODO: check if file has good form
    frames = f.read_frames(lengthinseconds*samplerate)
    sg = spectrogram.Spectrogram(frames, windowsize, stepsize, windowtype, fftres)
    return sg.spectrogram


def save_features_in_hdf5(features_path, ground_truth, spectrogram_options, tracks_options):
    h5file = tables.open_file(features_path, mode="w", title="Features")
    table = h5file.create_table("/", 'track', Track, "Track")
    tr = table.row
    for filename, genre in ground_truth.ground_truth.iteritems():
        tr['name'] = path.basename(filename)
        tr['path'] = filename
        tr['genre'] = genre
        tr['spectrogram'] = calc_spectrogram(filename, spectrogram_options, tracks_options)
        tr.append()
    h5file.close()

if __name__ == '__main__':
    conf = config.get_config()
    audio_folder = path.expanduser(conf.get('Input', 'AudioFolder'))
    extensions = conf.get('Input', 'AudioFileExtensions').split(' ')
    lists_folder = path.expanduser(conf.get('Preprocessing', 'ListsFolder'))
    ground_truth_path = path.join(lists_folder, 'ground_truth.pkl')
    features_folder = path.expanduser(conf.get('Preprocessing', 'FeaturesFolder'))
    features_path = path.join(features_folder, 'features.h5')
    if path.isdir(audio_folder):
        # ground truth
        audiofiles = list_audio_files_and_genres(audio_folder, extensions)
        gt = ground_truth.GroundTruth(audiofiles)
        gt.save_to_pickle_file(ground_truth_path)

        # feature extraction
        save_features_in_hdf5(
            features_path,
            gt,
            dict(conf.items('Spectrogram')),
            dict(conf.items('Tracks'))
        )

    else:
        logging.error("%s not found!" % audio_folder)
        sys.exit(1)
