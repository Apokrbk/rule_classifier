import pandas as pd
import matplotlib.pyplot as plt


def produce_diagram(files, line_labels, col_x, col_y, x_label, y_label, x_fontsize, y_fontsize, linewidth,
                    legend_fontsize=0, tick_size=0, x_size=10, y_size=10, groupby=False):
    df_files = list()
    lines = list()
    for i in range(0, len(files)):
        df_files.append(pd.read_csv(files[i], encoding='utf-8', delimiter=';'))
        if groupby == True:
            df_files[i] = df_files[i].groupby('Increasing').mean()
    plt.figure(figsize=(x_size, y_size))
    for i in range(0, len(files)):
        line, = plt.plot(df_files[i][col_x], df_files[i][col_y], linewidth=linewidth, label=line_labels[i])
    if legend_fontsize > 0:
        plt.legend(handles=lines)
        plt.legend(bbox_to_anchor=(1, 1), prop={'size': legend_fontsize})
    plt.xlabel(x_label, fontsize=x_fontsize)
    plt.ylabel(y_label, fontsize=y_fontsize)
    if tick_size > 0:
        plt.tick_params(labelsize=tick_size)
    plt.show()


produce_diagram(files=['results_files/mushroom_test.csv', 'results_files/mushroom_test2.csv'],
                line_labels=['bitmap','dict'],
                col_x='All train examples',
                col_y='Time in seconds',
                x_label='Liczba przykładów',
                y_label='Czas w sekundach',
                x_fontsize=20,
                y_fontsize=20,
                linewidth=2,
                legend_fontsize=15,
                tick_size=15,
                x_size=15,
                y_size=10,
                groupby=True
                )
