import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def produce_diagram(files, line_labels, col_x, col_y, x_label, y_label, x_fontsize, y_fontsize, linewidth,
                    legend_fontsize=0, tick_size=0, x_size=10, y_size=10, groupby=False):
    df_files = list()
    lines = list()
    for i in range(0, len(files)):
        df_files.append(pd.read_csv(files[i], encoding='utf-8', delimiter=';'))
        if groupby:
            df_files[i] = df_files[i].groupby('Increasing').mean()
    plt.figure(figsize=(x_size, y_size))
    for i in range(0, len(files)):
        line, = plt.plot(df_files[i][col_x], df_files[i][col_y], linewidth=linewidth, marker='o')
    if legend_fontsize > 0:
        plt.legend(handles=lines)
        plt.legend(bbox_to_anchor=(1, 1), prop={'size': legend_fontsize})
    plt.xlabel(x_label, fontsize=x_fontsize)
    plt.ylabel(y_label, fontsize=y_fontsize)
    if tick_size > 0:
        plt.tick_params(labelsize=tick_size)
    plt.show()


def cube_diagram(file1, file2, file3, file4, file5, file6, variable, x_size=12, y_size=8):
    df1 = pd.read_csv(file1, encoding='utf-8', delimiter=';')
    df1 = df1[variable]
    df2 = pd.read_csv(file2, encoding='utf-8', delimiter=';')
    df2 = df2[variable]
    df3 = pd.read_csv(file3, encoding='utf-8', delimiter=';')
    df3 = df3[variable]
    df4 = pd.read_csv(file4, encoding='utf-8', delimiter=';')
    df4 = df4[variable]
    df5 = pd.read_csv(file5, encoding='utf-8', delimiter=';')
    df5 = df5[variable]
    df6 = pd.read_csv(file6, encoding='utf-8', delimiter=';')
    df6 = df6[variable]
    data = [df1, df2, df3, df4, df5, df6]
    plt.boxplot(data, showfliers=False)
    plt.xticks([1, 2, 3, 4, 5, 6], ['DT', 'RF', 'RL', 'RD-A', 'RD-B', 'RD-BR'], fontsize=12)
    plt.xlabel('Metoda', fontsize=12)
    plt.ylabel('Liczba błędów', fontsize=12)
    plt.show()


def produce_diagram_for_one_file_with_errors(file, col_x, col_y_mean, col_y_std, x_label, y_label,
                                             x_fontsize, y_fontsize, linewidth, tick_size=0, x_size=15,
                                             y_size=10, groupby=False):
    plt.figure(figsize=(x_size, y_size))
    df = pd.read_csv(file, encoding='utf-8', delimiter=';')
    if groupby:
        df1 = df.groupby('Increasing').mean()
        df2 = df.groupby('Increasing').std()
        x = df1[col_x]
        y = df1[col_y_mean]
        e = df2[col_y_std]
    else:
        x = df[col_x]
        y = df[col_y_mean]
        e = df[col_y_std]
    plt.errorbar(x, y, yerr=e, linewidth=linewidth, fmt='-o', capsize=linewidth + 3)
    plt.xlabel(x_label, fontsize=x_fontsize)
    plt.ylabel(y_label, fontsize=y_fontsize)
    if tick_size > 0:
        plt.tick_params(labelsize=tick_size)
    plt.show()


