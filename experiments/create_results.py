import os

import pandas as pd


def get_mean_from_column_in_csv(file, column):
    df = pd.read_csv(file,
                     encoding='utf-8', delimiter=';')
    return round(df[column].mean(), 2)


def get_sum_from_column_in_csv(file, column):
    df = pd.read_csv(file,
                     encoding='utf-8', delimiter=';')
    return round(df[column].sum(), 2)


def get_std_dev_from_column_in_csv(file, column):
    df = pd.read_csv(file,
                     encoding='utf-8', delimiter=';')
    return round(df[column].std(), 2)


def get_mean_std_from_recall_in_csv(file):
    df = pd.read_csv(file,
                     encoding='utf-8', delimiter=';')
    df['recall'] = df['TP'] / (df['TP'] + df['FN'])
    return round(df['recall'].mean(), 2), round(df['recall'].std(), 2)


def get_mean_std_from_auc_in_csv(file):
    df = pd.read_csv(file,
                     encoding='utf-8', delimiter=';')
    df['recall'] = df['TP'] / (df['TP'] + df['FN'])
    df['specifity'] = df['TN'] / (df['TN'] + df['FP'])
    df['auc'] = df['recall'] * 0.5 + df['specifity'] * 0.5
    return round(df['auc'].mean(), 2), round(df['auc'].std(), 2)


def create_results_for_all_files_in_path_confusion_matrix(path, filename_r=''):
    files = list()
    tp_mean = list()
    tn_mean = list()
    fp_mean = list()
    fn_mean = list()
    for filename in os.listdir(path):
        files.append(filename)
        tp_mean.append(get_mean_from_column_in_csv(path + filename, 'TP'))
        tn_mean.append(get_mean_from_column_in_csv(path + filename, 'TN'))
        fp_mean.append(get_mean_from_column_in_csv(path + filename, 'FP'))
        fn_mean.append(get_mean_from_column_in_csv(path + filename, 'FN'))

    results = pd.DataFrame(
        {'filename': files,
         'TP': tp_mean,
         'TN': tn_mean,
         'FP': fp_mean,
         'FN': fn_mean
         })
    if filename_r == '':
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(results)
    else:
        results.to_csv(filename_r, sep=';', encoding='utf-8', index_label='id')


def create_results_for_all_files_in_path_errors_acc_recall_auc(path, filename_r=''):
    files = list()
    errors_mean = list()
    errors_std = list()
    acc_mean = list()
    acc_std = list()
    recall_mean = list()
    recall_std = list()
    auc_mean = list()
    auc_std = list()

    for filename in os.listdir(path):
        files.append(filename)
        print(filename)
        errors_mean.append(get_mean_from_column_in_csv(path + filename, 'Errors (FP + FN)'))
        errors_std.append(get_std_dev_from_column_in_csv(path + filename, 'Errors (FP + FN)'))
        acc_mean.append(get_mean_from_column_in_csv(path + filename, 'Accuracy'))
        acc_std.append(get_std_dev_from_column_in_csv(path + filename, 'Accuracy'))
        mean_temp, std_temp = get_mean_std_from_recall_in_csv(path + filename)
        recall_mean.append(mean_temp)
        recall_std.append(std_temp)
        mean_temp, std_temp = get_mean_std_from_auc_in_csv(path + filename)
        auc_mean.append(mean_temp)
        auc_std.append(std_temp)
    results = pd.DataFrame(
        {'filename': files,
         'err_mean': errors_mean,
         'err_std': errors_std,
         'acc_mean': acc_mean,
         'acc_std': acc_std,
         'rec_mean': recall_mean,
         'rec_std': recall_std,
         'auc_mean': auc_mean,
         'auc_std': auc_std
         })
    if filename_r == '':
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(results)
    else:
        results.to_csv(filename_r, sep=';', encoding='utf-8', index_label='id')


def create_results_for_all_files_in_path_time(path, filename_r=''):
    files = list()
    time_mean = list()
    time_std = list()
    features = list()
    all_train_examples = list()
    for filename in os.listdir(path):
        files.append(filename)
        time_mean.append(get_mean_from_column_in_csv(path + filename, 'Time in seconds'))
        time_std.append(get_std_dev_from_column_in_csv(path + filename, 'Time in seconds'))
        features.append(get_mean_from_column_in_csv(path + filename, 'Number of features'))
        all_train_examples.append(get_mean_from_column_in_csv(path + filename, 'All train examples'))
    results = pd.DataFrame(
        {'filename': files,
         'time_mean': time_mean,
         'time_std': time_std,
         'number_of_features': features,
         'train_examples': all_train_examples
         })
    if filename_r == '':
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(results)
    else:
        results.to_csv(filename_r, sep=';', encoding='utf-8', index_label='id')

def create_results_for_all_files_in_path_time_rules(path, filename_r=''):
    files = list()
    time_mean = list()
    time_std = list()
    features = list()
    all_train_examples = list()
    rules = list()
    for filename in os.listdir(path):
        files.append(filename)
        time_mean.append(get_mean_from_column_in_csv(path + filename, 'Time in seconds'))
        time_std.append(get_std_dev_from_column_in_csv(path + filename, 'Time in seconds'))
        features.append(get_mean_from_column_in_csv(path + filename, 'Number of features'))
        all_train_examples.append(get_mean_from_column_in_csv(path + filename, 'All train examples'))
        rules.append(get_mean_from_column_in_csv(path+filename, 'Number of rules'))
    results = pd.DataFrame(
        {'filename': files,
         'time_mean': time_mean,
         'time_std': time_std,
         'number_of_features': features,
         'train_examples': all_train_examples,
         'num_of_rules': rules
         })
    if filename_r == '':
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(results)
    else:
        results.to_csv(filename_r, sep=';', encoding='utf-8', index_label='id')


create_results_for_all_files_in_path_errors_acc_recall_auc("C:/Users/damia/Desktop/pracainz/rezultaty/random_rows/", '')