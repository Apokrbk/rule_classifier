import math
import pandas as pd
import time
from sklearn import tree
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from Classifier.abstract_datasets.bitmap_dataset.bitmap_dataset import BitmapDataset
from Classifier.abstract_datasets.dict_dataset.dict_dataset import DictDataset
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


def test_all(df_all, iters, filename, increasing, kfold, method, dataset_type=BitmapDataset, grow_param_raw=0,
             prune_param_raw=0):
    acc, all, all_train, errors, features, fn, fp, inc, kfold_list, n, number_of_rules, p, times, tn, tp = init_results()
    for j in range(0, increasing):
        for i in range(0, iters):
            kfold_datasets = split_kfold(df_all, kfold)
            for k in range(0, kfold):
                df_test, df_train = divide_into_test_train(k, kfold_datasets)
                count_examples(all, all_train, df_test, df_train, n, p)
                X_train = df_train.loc[:, df_train.columns != df_train.columns[-1]]
                Y_train = df_train[df_train.columns[-1]]
                X_test = df_test.loc[:, df_test.columns != df_test.columns[-1]]
                Y_test = df_test[df_test.columns[-1]]
                fn_tmp, fp_tmp, tn_tmp, tp_tmp, time_tmp, number_of_rules_tmp = method(X_train, Y_train, X_test, Y_test,
                                                                                       dataset_type=dataset_type,
                                                                                       grow_param_raw=grow_param_raw,
                                                                                       prune_param_raw=prune_param_raw)
                add_results(acc, df_train, errors, features, fn, fn_tmp, fp, fp_tmp, inc, j, k, kfold_list, tn, tn_tmp,
                            tp, tp_tmp, times, time_tmp, number_of_rules, number_of_rules_tmp)
                print("Iter: " + str(i))
        df_all = pd.concat([df_all] * 2, ignore_index=True)
        print("Increasing: " + str(j))
    results = create_results(acc, all, all_train, errors, features, fn, fp, inc, kfold_list, n, number_of_rules, p,
                             times, tn, tp)
    if filename != '':
        results.to_csv(filename, sep=';', encoding='utf-8', index_label='id')
    else:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(results)
    return results


def create_results(acc, all, all_train, errors, features, fn, fp, inc, kfold_list, n, number_of_rules, p, times, tn,
                   tp):
    return pd.DataFrame(
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
         'Increasing': inc
         })


def add_results(acc, df_train, errors, features, fn, fn_tmp, fp, fp_tmp, inc, j, k, kfold_list, tn, tn_tmp, tp, tp_tmp,
                times, time_tmp, number_of_rules, number_of_rules_tmp):
    times.append(time_tmp)
    number_of_rules.append(number_of_rules_tmp)
    tp.append(tp_tmp)
    fp.append(fp_tmp)
    tn.append(tn_tmp)
    fn.append(fn_tmp)
    kfold_list.append(k + 1)
    errors.append(fp_tmp + fn_tmp)
    features.append(len(df_train.columns) - 1)
    acc.append((tp_tmp + tn_tmp) / (tp_tmp + tn_tmp + fp_tmp + fn_tmp))
    inc.append(j)


def count_examples(all, all_train, df_test, df_train, n, p):
    all_ex = len(df_test)
    p_ex = df_test[df_test.columns[-1]].sum()
    n_ex = all_ex - p_ex
    all_train_ex = len(df_train)
    all_train.append(all_train_ex)
    p.append(p_ex)
    n.append(n_ex)
    all.append(all_ex)


def divide_into_test_train(k, kfold_datasets):
    df_test = kfold_datasets[k]
    df_train = pd.concat(exclude(kfold_datasets, k))
    df_test.index = range(len(df_test))
    df_train.index = range(len(df_train))
    return df_test, df_train


def init_results():
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
    return acc, all, all_train, errors, features, fn, fp, inc, kfold_list, n, number_of_rules, p, times, tn, tp


def test_rule_creator(X_train, Y_train, X_test, Y_test, dataset_type=BitmapDataset,
                      grow_param_raw=0, prune_param_raw=0):

    start = time.time()
    rule_creator = RuleCreator(dataset_type=dataset_type, grow_param_raw=grow_param_raw,
                               prune_param_raw=prune_param_raw)
    rule_creator.fit(X_train, Y_train)
    end = time.time()
    predictions = rule_creator.predict(X_test)
    tp_tmp, fp_tmp, tn_tmp, fn_tmp = count_tp_fp_tn_fn(Y_test,predictions)
    return fn_tmp, fp_tmp, tn_tmp, tp_tmp, end-start, rule_creator.get_number_of_rules()


def test_regression(X_train, Y_train, X_test, Y_test, dataset_type=BitmapDataset, grow_param_raw=0, prune_param_raw=0):
    start = time.time()
    X_test, X_train, Y_test, Y_train = preprocess_for_scikit(X_test, X_train, Y_test, Y_train)
    clf = LogisticRegression()
    clf = clf.fit(X_train, Y_train)
    end = time.time()
    predictions = clf.predict(X_test)
    tp_tmp, fp_tmp, tn_tmp, fn_tmp = count_tp_fp_tn_fn(Y_test, predictions)
    return fn_tmp, fp_tmp, tn_tmp, tp_tmp, end - start, -1


def preprocess_for_scikit(X_test, X_train, Y_test, Y_train):
    le = preprocessing.LabelEncoder()
    X_train = X_train.apply(le.fit_transform)
    X_test = X_test.apply(le.fit_transform)
    return X_test, X_train, Y_test, Y_train


def test_random_forest(X_train, Y_train, X_test, Y_test, dataset_type=BitmapDataset,
                       grow_param_raw=0, prune_param_raw=0):
    start = time.time()
    X_test, X_train, Y_test, Y_train = preprocess_for_scikit(X_test, X_train, Y_test, Y_train)
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf = clf.fit(X_train, Y_train)
    end = time.time()
    predictions = clf.predict(X_test)
    tp_tmp, fp_tmp, tn_tmp, fn_tmp = count_tp_fp_tn_fn(Y_test, predictions)
    return fn_tmp, fp_tmp, tn_tmp, tp_tmp, end - start, -1


def test_tree(X_train, Y_train, X_test, Y_test, dataset_type=BitmapDataset, grow_param_raw=0,
              prune_param_raw=0):
    start = time.time()
    X_test, X_train, Y_test, Y_train = preprocess_for_scikit(X_test, X_train, Y_test, Y_train)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)
    end = time.time()
    predictions = clf.predict(X_test)
    tp_tmp, fp_tmp, tn_tmp, fn_tmp = count_tp_fp_tn_fn(Y_test, predictions)
    return fn_tmp, fp_tmp, tn_tmp, tp_tmp, end-start, -1




df = pd.read_csv('data_files/mushroom.csv',
                 encoding='utf-8', delimiter=';')
test_all(df, 1, 'results_files/mushroom_rule_creator.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset)
test_all(df, 1, 'results_files/mushroom_tree.csv', 1, 5, method=test_tree)
test_all(df, 1, 'results_files/mushroom_random_forest.csv', 1, 5, method=test_random_forest)
test_all(df, 1, 'results_files/mushroom_regression.csv', 1, 5, method=test_regression)
df = cubes_for_numeric_data(df, 10)
test_all(df, 1, 'results_files/mushroom_rule_creator.csv', 1, 5, method=test_rule_creator)

df = pd.read_csv('data_files/phoneme.csv',
                 encoding='utf-8', delimiter=';')

test_all(df, 1, 'results_files/phoneme_rule_creator.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset)
test_all(df, 1, 'results_files/phoneme_tree.csv', 1, 5, method=test_tree)
test_all(df, 1, 'results_files/phoneme_random_forest.csv', 1, 5, method=test_random_forest)
test_all(df, 1, 'results_files/phoneme_regression.csv', 1, 5, method=test_regression)
df = cubes_for_numeric_data(df, 10)
test_all(df, 1, 'results_files/phoneme_rule_creator.csv', 1, 5, method=test_rule_creator)

df = pd.read_csv('data_files/hypothyroid.csv',
                 encoding='utf-8', delimiter=';')

test_all(df, 1, 'results_files/hypothyroid_rule_creator.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset)
test_all(df, 1, 'results_files/hypothyroid_tree.csv', 1, 5, method=test_tree)
test_all(df, 1, 'results_files/hypothyroid_random_forest.csv', 1, 5, method=test_random_forest)
test_all(df, 1, 'results_files/hypothyroid_regression.csv', 1, 5, method=test_regression)
df = cubes_for_numeric_data(df, 10)
test_all(df, 1, 'results_files/hypothyroid_rule_creator.csv', 1, 5, method=test_rule_creator)

df = pd.read_csv('data_files/glass.csv',
                 encoding='utf-8', delimiter=';')

test_all(df, 1, 'results_files/glass_rule_creator.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset)
test_all(df, 1, 'results_files/glass_tree.csv', 1, 5, method=test_tree)
test_all(df, 1, 'results_files/glass_random_forest.csv', 1, 5, method=test_random_forest)
test_all(df, 1, 'results_files/glass_regression.csv', 1, 5, method=test_regression)
df = cubes_for_numeric_data(df, 10)
test_all(df, 1, 'results_files/glass_rule_creator.csv', 1, 5, method=test_rule_creator)

