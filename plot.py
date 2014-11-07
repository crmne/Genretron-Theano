import os
import config
import utils

conf = config.get_config()
save_plot = True if conf.get('Output', 'SavePlot') == 'Yes' else False
show_plot = True if conf.get('Output', 'ShowPlot') == 'Yes' else False


def import_global(module_name):
    globals()[module_name] = __import__(module_name)

if save_plot or show_plot:
    import_global('matplotlib')
    if not show_plot:
        matplotlib.use("Agg")
    import_global('matplotlib.pyplot')


class Plot(object):
    def __init__(self, *what_to_plot):
        for x in what_to_plot:
            assert isinstance(x, str)
        self.data = {key: [] for key in what_to_plot}
        self.length = 0
        if show_plot:
            self.fig = matplotlib.pyplot.figure()
            matplotlib.pyplot.ion()
            matplotlib.pyplot.show()

    def append(self, key, value):
        if key not in self.data:
            raise StandardError("Plot: Key %s not found." % key)
        self.data[key].append(value)
        self.length = max([len(v) for k, v in self.data.iteritems()])

    def update_plot(self):
        if save_plot or show_plot:
            x = self.length
            dim_x, dim_y = utils.find_two_closest_factors(len(self.data))

            i = 1
            for k, v in self.data.iteritems():
                if len(v) != 0:
                    matplotlib.pyplot.subplot(dim_x, dim_y, i)
                    # FIXME: deal with NaNs
                    matplotlib.pyplot.plot(x, v[-1], 'r.-')
                    matplotlib.pyplot.title(k)
                i = i + 1

            if show_plot:
                matplotlib.pyplot.draw()

    def save_plot(self, format='PDF'):
        output_folder = os.path.expanduser(conf.get('Output', 'OutputFolder'))
        run_n = int(conf.get('CrossValidation', 'RunNumber'))
        output_file = os.path.join(output_folder, 'plot%i.%s' % (run_n, format))
        self.update_plot()
        if save_plot:
            import logging
            # logging.debug("Plot saved in %s" % output_file)
            matplotlib.pyplot.savefig(output_file, format=format)
