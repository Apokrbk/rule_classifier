import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
        line, = plt.plot(df_files[i][col_x], df_files[i][col_y], linewidth=linewidth, marker='o')
    if legend_fontsize > 0:
        plt.legend(handles=lines)
        plt.legend(bbox_to_anchor=(1, 1), prop={'size': legend_fontsize})
    plt.xlabel(x_label, fontsize=x_fontsize)
    plt.ylabel(y_label, fontsize=y_fontsize)
    if tick_size > 0:
        plt.tick_params(labelsize=tick_size)
    plt.show()


# produce_diagram(files=['old_used/mushjroom_rule_creator_bitmap_inc.csv'],
#                 line_labels=['bitmap'],
#                 col_x='All train examples',
#                 col_y='Time in seconds',
#                 x_label='Liczba przykładów',
#                 y_label='Czas w sekundach',
#                 x_fontsize=20,
#                 y_fontsize=20,
#                 linewidth=2,
#                 legend_fontsize=15,
#                 tick_size=15,
#                 x_size=12,
#                 y_size=8,
#                 groupby=True
#                 )

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
    plt.xticks([1, 2, 3, 4 , 5, 6], ['DT', 'RF', 'RL', 'RD-A', 'RD-B', 'RD-BR'], fontsize=12)
    plt.xlabel('Metoda', fontsize=12)
    plt.ylabel('Liczba błędów', fontsize=12)
    plt.show()


cube_diagram('old_used/income_tree.csv',
             'old_used/income_random_forest_100trees.csv',
             'old_used/income_regression.csv',
             'old_used/income_rule_creator_dict_0_0.csv',
             'old_used/income_rule_creator_bitmap_10_0_0.csv',
             'old_used/income_rule_creator_bitmap_50_0_0_R.csv',
             'Errors (FP + FN)')


cube_diagram('old_used/nba_5y_tree.csv',
             'old_used/nba_5y_random_forest_100trees.csv',
             'old_used/nba_5y_regression.csv',
             'old_used/nba_5y_rule_creator_dict_01_0.csv',
             'old_used/nba_5y_rule_creator_bitmap_10_005_005.csv',
             'old_used/nba_5y_rule_creator_bitmap_10_005_005_R.csv',
             'Errors (FP + FN)')


cube_diagram('old_used/phoneme_tree.csv',
             'old_used/phoneme_random_forest_100trees.csv',
             'old_used/phoneme_regression.csv',
             'old_used/phoneme_rule_creator_dict_005_005.csv',
             'old_used/phoneme_rule_creator_bitmap_10_0_0.csv',
             'old_used/phoneme_rule_creator_bitmap_10_0_0_R.csv',
             'Errors (FP + FN)')


cube_diagram('old_used/glass_tree.csv',
             'old_used/glass_random_forest_100trees.csv',
             'old_used/glass_regression.csv',
             'old_used/glass_rule_creator_dict_03_03.csv',
             'old_used/glass_rule_creator_bitmap_10_02_02.csv',
             'old_used/glass_rule_creator_bitmap_10_02_02_R.csv',
             'Errors (FP + FN)')


cube_diagram('old_used/breast_cancer_tree.csv',
             'old_used/breast_cancer_random_forest_100trees.csv',
             'old_used/breast_cancer_regression.csv',
             'old_used/breast_cancer_rule_creator_dict_0_0.csv',
             'old_used/breast_cancer_rule_creator_bitmap_10_005_005.csv',
             'old_used/breast_cancer_rule_creator_bitmap_10_0_0_R.csv',
             'Errors (FP + FN)')


cube_diagram('old_used/mushroom_tree.csv',
             'old_used/mushroom_random_forest_100trees.csv',
             'old_used/mushroom_regression.csv',
             'old_used/mushroom_rule_creator_dict.csv',
             'old_used/mushroom_rule_creator_bitmap.csv',
             'old_used/mushroom_rule_creator_bitmap_R.csv',
             'Errors (FP + FN)')