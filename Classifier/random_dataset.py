import random
import string
import pandas as pd
import numpy as np
from Classifier.test_all_results import test_all, test_rule_creator


def create_random_dataframe(num_of_features, num_of_rows, condition, prob):
    alphabet = list(string.ascii_lowercase)
    columns = list()
    col_names = list()
    for j in range(0, num_of_features):
        column = list()
        for i in range(0, num_of_rows):
            column.append(alphabet[random.randint(0, len(alphabet) - 1)])
        columns.append(column)
        col_names.append('a' + str(j))
    class_col = list()
    for i in range(0, num_of_rows):
        if condition(columns, i) and random.random() > prob:
            class_col.append(1)
        else:
            if random.random() > prob:
                class_col.append(0)
            else:
                class_col.append(1)
    columns.append(class_col)
    col_names.append('Class')
    df = pd.concat([pd.Series(x) for x in columns], axis=1)
    df.columns = col_names
    return df


def condition1(columns, i):
    if columns[0][i] in ('a', 'b', 'c', 'd', 'e'):
        return True
    elif columns[0][i] in ('j', 'k', 'l', 'm', 'n', 'o', 'p') and columns[1][i] in ('a', 'b', 'c', 'd', 'e', 'f'):
        return True
    elif columns[2][i] in ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k') and columns[3][i] in ('x', 'y', 'z'):
        return True
    elif columns[4][i] in ('a', 'b', 'c', 'd', 'e'):
        return True
    else:
        return False


for i in range(20, 25):
    df = create_random_dataframe(5 + i * 2, 100000, condition1, 0.1)
    test_all(df, 5, 'results_files/random_dataset_' + str(i)+'.csv', 1, 5, method=test_rule_creator)
