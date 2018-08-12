from pyroaring import BitMap

import pandas as pd

from Classifier.BitmapDataset.BitmapDataset import BitmapDataset

df = pd.read_csv('C:/Users/damia/Desktop/pracainz/dane/dane_testowe/test_literal_count_p_n_1.csv',
                         encoding='utf-8', delimiter=',')
x = BitMap()
x.add(2)
x.add(5)
print(len(x))


