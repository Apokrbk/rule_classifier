import math
import pandas as pd
import time
from sklearn import tree
from sklearn import preprocessing

from Classifier.abstract_datasets.bitmap_dataset.bitmap_dataset import BitmapDataset
from Classifier.abstract_datasets.dict_dataset.dict_dataset import DictDataset
from Classifier.abstract_datasets.nparray_dataset.nparray_dataset import NpArrayDataset
from Classifier.rule_creator import RuleCreator


def split_kfold(df, kfold):
    df = df.sample(frac=1)
    df.index = range(len(df))
    div_idx = math.floor(len(df) / kfold)
    kfold_sets = list()
    for i in range(0, kfold):
        if i == kfold - 1:
            temp_set = df[i * div_idx:]
        else:
            temp_set = df[i * div_idx:(i + 1) * div_idx]
        kfold_sets.append(temp_set)
    return kfold_sets


def cubes_for_numeric_data(df, num_of_intervals):
    numeric_cols = df._get_numeric_data().columns
    numeric_cols = numeric_cols[:-1]
    for i in range(0, len(numeric_cols)):
        df[numeric_cols[i]] = pd.qcut(df[numeric_cols[i]], num_of_intervals, duplicates='drop').astype(str)
    return df


def remove_empty_values(df):
    numeric_cols = df._get_numeric_data().columns
    char_cols = df.select_dtypes('object').columns
    for i in range(0, len(numeric_cols)):
        avg = df[numeric_cols[i]].mean()
        df[numeric_cols[i]] = df[numeric_cols[i]].fillna(value=avg)
    for i in range(0, len(char_cols)):
        df[char_cols[i]] = df[char_cols[i]].fillna(value='null')
    return df


def count_tp_fp_tn_fn(classes, predictions):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    if len(classes) != len(predictions):
        print("Vector with classes and vector with predictions should have same lenght")
        return
    for i in range(0, len(classes)):
        if classes[i] == predictions[i]:
            if classes[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if classes[i] == 1:
                fn += 1
            else:
                fp += 1
    return tp, fp, tn, fn


def exclude(lst, i):
    if i == 0:
        return lst[i + 1:]
    return lst[:i] + lst[i + 1:]


def test_all(df_all, iters, filename, increasing, kfold, dataset_type=None, dtree=False):
    tp = list()
    fp = list()
    tn = list()
    fn = list()
    times = list()
    number_of_rules = list()
    errors = list()
    p = list()
    n = list()
    all = list()
    features = list()
    all_train = list()
    acc = list()
    kfold_list = list()
    inc = list()
    for j in range(0, increasing):
        for i in range(0, iters):
            kfold_datasets = split_kfold(df_all, kfold)
            for k in range(0, kfold):
                df_test = kfold_datasets[k]
                df_train = pd.concat(exclude(kfold_datasets, k))
                df_test.index = range(len(df_test))
                df_train.index = range(len(df_train))
                print(len(df_test))
                print(len(df_train))
                all_ex = len(df_test)
                p_ex = df_test[df_test.columns[-1]].sum()
                n_ex = all_ex - p_ex
                all_train_ex = len(df_train)
                all_train.append(all_train_ex)
                p.append(p_ex)
                n.append(n_ex)
                all.append(all_ex)
                if dtree == True:
                    start = time.time()
                    le = preprocessing.LabelEncoder()
                    df_train = df_train.apply(le.fit_transform)
                    df_test = df_test.apply(le.fit_transform)
                    X = df_train.loc[:, df_train.columns != df_train.columns[-1]]
                    Y = df_train[df_train.columns[-1]]
                    clf = tree.DecisionTreeClassifier()
                    clf = clf.fit(X, Y)
                    end = time.time()
                    predictions = clf.predict(df_test.loc[:, df_test.columns != df_test.columns[-1]])
                    number_of_rules.append(-1)
                    tp_tmp, fp_tmp, tn_tmp, fn_tmp = count_tp_fp_tn_fn(df_test[df_test.columns[-1]], predictions)
                else:
                    start = time.time()
                    rule_creator = RuleCreator(dataset_type, df_train, 1)
                    end = time.time()
                    number_of_rules.append(rule_creator.get_number_of_rules())
                    predictions = rule_creator.predict(df_test)
                    tp_tmp, fp_tmp, tn_tmp, fn_tmp = count_tp_fp_tn_fn(df_test[df_test.columns[-1]].tolist(),
                                                                       predictions)
                times.append(end - start)
                tp.append(tp_tmp)
                fp.append(fp_tmp)
                tn.append(tn_tmp)
                fn.append(fn_tmp)
                kfold_list.append(k + 1)
                errors.append(fp_tmp + fn_tmp)
                features.append(len(df_train.columns) - 1)
                acc.append((tp_tmp + tn_tmp) / (tp_tmp + tn_tmp + fp_tmp + fn_tmp))
                inc.append(j)
                print("Iter: " + str(i))
        df_all = pd.concat([df_all] * 2, ignore_index=True)
        print("Increasing: " + str(j))
    results = pd.DataFrame(
        {'All train examples': all_train,
         'All test examples': all,
         'Positive examples': p,
         'Negative examples': n,
         'TP': tp,
         'FP': fp,
         'TN': tn,
         'FN': fn,
         'Errors (FP + FN)': errors,
         'Accuracy': acc,
         'Time in seconds': times,
         'Number of rules': number_of_rules,
         'Number of features': features,
         'Kfold': kfold_list,
         'Increasing' : inc
         })
    if filename != '':
        results.to_csv(filename, sep=';', encoding='utf-8', index_label='id')
    else:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(results)
    return results


df = pd.read_csv('data_files/mushroom.csv',
                 encoding='utf-8', delimiter=';')

# df = cubes_for_numeric_data(df,5)
test_all(df, 3, 'results_files/mushroom124215.csv', 1, 5, dataset_type=DictDataset, dtree=False)

