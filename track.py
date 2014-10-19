from tables import *
import config
import utils
from spectrogram import Spectrogram

conf = config.get_config()
winsize = int(conf.get('Spectrogram', 'WindowSize'))
nframes = int(conf.get('Tracks', 'LengthInSeconds')) * \
    int(conf.get('Tracks', 'SampleRate'))
stepsize = int(conf.get('Spectrogram', 'StepSize'))
fftres = int(conf.get('Spectrogram', 'FFTResolution'))
audio_folder = os.path.expanduser(conf.get('Input', 'AudioFolder'))
numberofgenres = len(utils.list_subdirs(audio_folder))
wins = Spectrogram.wins(winsize, nframes, stepsize)
bins = Spectrogram.bins(fftres)
shape = Spectrogram.shape(wins, bins)


class Track(IsDescription):

    """Description of a track in HDF5"""
    idnumber = Int32Col()
    name = StringCol(64)
    path = StringCol(512)
    genre = StringCol(32)
    # target = BoolCol(shape=(numberofgenres,))
    target = Int8Col()
    spectrogram = Float32Col(dflt=0.0, shape=shape)
