from tables import *
import config
from spectrogram import Spectrogram

conf = config.get_config()
winsize = int(conf.get('Spectrogram', 'WindowSize'))
nframes = int(conf.get('Tracks', 'LengthInSeconds')) * int(conf.get('Tracks', 'SampleRate'))
stepsize = int(conf.get('Spectrogram', 'StepSize'))
fftres = int(conf.get('Spectrogram', 'FFTResolution'))
wins = Spectrogram.wins(winsize, nframes, stepsize)
bins = Spectrogram.bins(fftres)
shape = Spectrogram.shape(wins, bins)


class Track(IsDescription):
    """Description of a track in HDF5"""
    idnumber = Int64Col()
    name = StringCol(64)
    path = StringCol(512)
    genre = StringCol(32)
    # TODO: get those value from somewhere else
    spectrogram = Float32Col(dflt=0.0, shape=shape)
