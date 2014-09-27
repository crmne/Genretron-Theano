import numpy


class Spectrogram(object):
    windows = {
        'square': numpy.ones,
        'hamming': numpy.hamming,
        'hanning': numpy.hanning,
        'bartlett': numpy.bartlett,
        'blackman': numpy.blackman
    }

    @staticmethod
    def shape(wins, bins):
        return len(wins), bins

    @staticmethod
    def wins(win_size, nframes, step_size):
        return range(win_size, nframes, step_size)

    @staticmethod
    def bins(fft_resolution):
        return fft_resolution/2 + 1

    def __init__(self, frames, win_size, step_size, window_type, fft_resolution):
        window = Spectrogram.windows[window_type](win_size)

        # will take windows x[n1:n2].  generate
        # and loop over n2 such that all frames
        # fit within the waveform
        wins = Spectrogram.wins(win_size, len(frames), step_size)
        bins = Spectrogram.bins(fft_resolution)

        self.spectrogram = numpy.zeros(Spectrogram.shape(wins, bins))

        for i, n in enumerate(wins):
            xseg = frames[n-win_size:n]
            z = numpy.fft.fft(window * xseg, fft_resolution)
            self.spectrogram[i, :] = numpy.log(numpy.abs(z[:bins]))

    def plot(self):
        import matplotlib.pyplot as plt
        plt.imshow(
            self.spectrogram.T,
            interpolation='nearest',
            origin='lower',
            aspect='auto'
        )
        plt.show()
