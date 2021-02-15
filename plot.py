
import matplotlib.pyplot as plt
from cycler import cycler

class Plot:
    def __init__(self, title):
        plt.title(title)

    def display(self):
        # show a legend on the plot
        plt.legend()
        # Display a figure.
        plt.show()

    def plot_line(self, plot_data_dicts, color):
        x=6

    def plot_points(self, plot_data_dicts, colors_list):
        plt.rc('axes', prop_cycle=(cycler('color', colors_list)))

        dicts_iter=iter(plot_data_dicts.items())
        xlabel,xvalues=next(dicts_iter)
        y1label,y1values=next(dicts_iter)
        y2label,y2values=next(dicts_iter)
        plt.xlabel(xlabel)
        plt.ylabel(y1label+ " and " + y2label)


        plt.plot(xvalues, y1values, label=y1label)
        plt.plot(xvalues, y2values, label=y2label)
