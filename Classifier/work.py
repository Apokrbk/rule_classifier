import os

import pandas as pd


def get_mean(file, column):
    df = pd.read_csv(file,
                     encoding='utf-8', delimiter=';')
    return round(df[column].mean(), 2)


def get_std(file, column):
    df = pd.read_csv(file,
                     encoding='utf-8', delimiter=';')
    return round(df[column].std(), 2)


def get_mean_recall(file):
    df = pd.read_csv(file,
                     encoding='utf-8', delimiter=';')
    df['recall'] = df['TP'] / (df['TP'] + df['FN'])
    return round(df['recall'].mean(), 2), round(df['recall'].std(), 2)

# get_all('results_files/phoneme_rule_creator_bitmap_10_03_03.csv')




def create_results(filename_r=''):
    files = list()
    times_mean = list()
    times_std = list()

    path='results_files/'
    for filename in os.listdir(path):
        files.append(filename)
        times_mean.append(get_mean(path + filename,'Time in seconds'))
        times_std.append(get_std(path + filename,'Time in seconds'))

    results = pd.DataFrame(
        {'filename': files,
         'time_mean': times_mean,
         'time_std': times_std,
         })
    if filename_r=='':
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(results)
    else:
        results.to_csv(filename_r, sep=';', encoding='utf-8', index_label='id')

create_results('jakosciowe/test2255.csv')
