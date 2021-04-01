import matplotlib.pyplot as plt


class Plot:
    xvalues = None

    def __init__(self, title, colors_list, x_dict, y_label):
        plt.title(title)
        plt.gca().set_prop_cycle('color', colors_list)
        x_label, self.xvalues = next(iter(x_dict.items()))
        plt.xlabel(x_label)
        plt.ylabel(y_label)

    def display(self):
        # show a legend on the plot
        plt.legend()
        # Display a figure.
        plt.show()

    def plot_line(self, y_label, y_values):
        plt.plot(self.xvalues, y_values, label=y_label)
        plt.ylabel(self.y_labels)

    def plot_lines(self, plot_data_dicts):
        dicts_iter = iter(plot_data_dicts.items())

        y_label, y_values = next(dicts_iter, (None, None))
        self.y_labels = y_label
        while y_label:
            self.plot_line(y_label, y_values)
            y_label, y_values = next(dicts_iter, (None, None))
