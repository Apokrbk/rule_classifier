from experiments.random_dataset import create_random_dataframe, condition1
from experiments.test_all_results import test_rule_creator, test_all, test_tree, test_random_forest, test_regression, \
    cubes_for_numeric_data
from rule_induction_classifier.rule_induction_classifier.abstract_datasets.bitmap_dataset.bitmap_dataset import BitmapDataset
from rule_induction_classifier.rule_induction_classifier.abstract_datasets.dict_dataset.dict_dataset import DictDataset
import pandas as pd
#
# print("MUSHROOM")
# df = pd.read_csv('data_files/mushroom.csv',
#                  encoding='utf-8', delimiter=';')
# test_all(df, 10, 'results_files/mushroom_rule_creator_bitmap.csv', 1, 5, method=test_rule_creator, roulette_selection=True)
# test_all(df, 10, 'results_files/mushroom_rule_creator_dict.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset)
# test_all(df, 10, 'results_files/mushroom_tree.csv', 1, 5, method=test_tree)
# test_all(df, 10, 'results_files/mushroom_random_forest_10trees.csv', 1, 5, method=test_random_forest)
# test_all(df, 10, 'results_files/mushroom_regression.csv', 1, 5, method=test_regression)
# test_all(df, 10, 'results_files/mushroom_rule_creator_bitmap.csv', 1, 5, method=test_rule_creator)
#
# print("PHONEME")
# df = pd.read_csv('data_files/phoneme.csv',
#                  encoding='utf-8', delimiter=';')
# test_all(df, 10, 'results_files/phoneme_tree.csv', 1, 5, method=test_tree)
# test_all(df, 10, 'results_files/phoneme_random_forest_10trees.csv', 1, 5, method=test_random_forest)
# test_all(df, 10, 'results_files/phoneme_random_forest_100trees.csv', 1, 5, method=test_random_forest, forest_trees=100)
# test_all(df, 10, 'results_files/phoneme_regression.csv', 1, 5, method=test_regression)
#
# test_all(df, 10, 'results_files/phoneme_rule_creator_dict_0_0.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset)
# test_all(df, 10, 'results_files/phoneme_rule_creator_dict_01_0.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.1)
# test_all(df, 10, 'results_files/phoneme_rule_creator_dict_0_01.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, prune_param_raw=0.1)
# test_all(df, 10, 'results_files/phoneme_rule_creator_dict_005_005.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.05, prune_param_raw=0.05)
# test_all(df, 10, 'results_files/phoneme_rule_creator_dict_02_02.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.2, prune_param_raw=0.2)
# test_all(df, 10, 'results_files/phoneme_rule_creator_dict_03_03.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.3, prune_param_raw=0.3)
# df = cubes_for_numeric_data(df, 10)
# test_all(df, 10, 'results_files/phoneme_rule_creator_bitmap_10_0_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, roulette_selection=True)
# test_all(df, 10, 'results_files/phoneme_rule_creator_bitmap_10_01_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/phoneme_rule_creator_bitmap_10_0_01_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, prune_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/phoneme_rule_creator_bitmap_10_005_005_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.05, prune_param_raw=0.05, roulette_selection=True)
# test_all(df, 10, 'results_files/phoneme_rule_creator_bitmap_10_02_02_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.2, prune_param_raw=0.2, roulette_selection=True)
# test_all(df, 10, 'results_files/phoneme_rule_creator_bitmap_10_03_03_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.3, prune_param_raw=0.3, roulette_selection=True)
# df = pd.read_csv('data_files/phoneme.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,20)
# test_all(df, 10, 'results_files/phoneme_rule_creator_bitmap_20_0_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, roulette_selection=True)
# test_all(df, 10, 'results_files/phoneme_rule_creator_bitmap_20_01_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/phoneme_rule_creator_bitmap_20_0_01_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, prune_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/phoneme_rule_creator_bitmap_20_005_005_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.05, prune_param_raw=0.05, roulette_selection=True)
# test_all(df, 10, 'results_files/phoneme_rule_creator_bitmap_20_02_02_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.2, prune_param_raw=0.2, roulette_selection=True)
# test_all(df, 10, 'results_files/phoneme_rule_creator_bitmap_20_03_03_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.3, prune_param_raw=0.3, roulette_selection=True)
#
# df = pd.read_csv('data_files/phoneme.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,50)
# test_all(df, 10, 'results_files/phoneme_rule_creator_bitmap_50_0_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, roulette_selection=True)
# test_all(df, 10, 'results_files/phoneme_rule_creator_bitmap_50_01_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/phoneme_rule_creator_bitmap_50_0_01_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, prune_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/phoneme_rule_creator_bitmap_50_005_005_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.05, prune_param_raw=0.05, roulette_selection=True)
# test_all(df, 10, 'results_files/phoneme_rule_creator_bitmap_50_02_02_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.2, prune_param_raw=0.2, roulette_selection=True)
# test_all(df, 10, 'results_files/phoneme_rule_creator_bitmap_50_03_03_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.3, prune_param_raw=0.3, roulette_selection=True)
# #
# print("GLASS")
# df = pd.read_csv('data_files/glass.csv',
#                  encoding='utf-8', delimiter=';')
# test_all(df, 10, 'results_files/glass_tree.csv', 1, 5, method=test_tree)
# test_all(df, 10, 'results_files/glass_random_forest_10trees.csv', 1, 5, method=test_random_forest)
# test_all(df, 10, 'results_files/glass_random_forest_100trees.csv', 1, 5, method=test_random_forest, forest_trees=100)
# test_all(df, 10, 'results_files/glass_regression.csv', 1, 5, method=test_regression)
#
# test_all(df, 10, 'results_files/glass_rule_creator_dict_0_0.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset)
# test_all(df, 10, 'results_files/glass_rule_creator_dict_01_0.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.1)
# test_all(df, 10, 'results_files/glass_rule_creator_dict_0_01.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, prune_param_raw=0.1)
# test_all(df, 10, 'results_files/glass_rule_creator_dict_005_005.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.05, prune_param_raw=0.05)
# test_all(df, 10, 'results_files/glass_rule_creator_dict_02_02.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.2, prune_param_raw=0.2)
# test_all(df, 10, 'results_files/glass_rule_creator_dict_03_03.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.3, prune_param_raw=0.3)
# df = cubes_for_numeric_data(df, 10)
# test_all(df, 10, 'results_files/glass_rule_creator_bitmap_10_0_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, roulette_selection=True)
# test_all(df, 10, 'results_files/glass_rule_creator_bitmap_10_01_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/glass_rule_creator_bitmap_10_0_01_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, prune_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/glass_rule_creator_bitmap_10_005_005_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.05, prune_param_raw=0.05, roulette_selection=True)
# test_all(df, 10, 'results_files/glass_rule_creator_bitmap_10_02_02_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.2, prune_param_raw=0.2, roulette_selection=True)
# test_all(df, 10, 'results_files/glass_rule_creator_bitmap_10_03_03_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.3, prune_param_raw=0.3, roulette_selection=True)
# df = pd.read_csv('data_files/glass.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,20)
# test_all(df, 10, 'results_files/glass_rule_creator_bitmap_20_0_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, roulette_selection=True)
# test_all(df, 10, 'results_files/glass_rule_creator_bitmap_20_01_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/glass_rule_creator_bitmap_20_0_01_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, prune_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/glass_rule_creator_bitmap_20_005_005_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.05, prune_param_raw=0.05, roulette_selection=True)
# test_all(df, 10, 'results_files/glass_rule_creator_bitmap_20_02_02_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.2, prune_param_raw=0.2, roulette_selection=True)
# test_all(df, 10, 'results_files/glass_rule_creator_bitmap_20_03_03_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.3, prune_param_raw=0.3, roulette_selection=True)
#
# df = pd.read_csv('data_files/glass.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,50)
# test_all(df, 10, 'results_files/glass_rule_creator_bitmap_50_0_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, roulette_selection=True)
# test_all(df, 10, 'results_files/glass_rule_creator_bitmap_50_01_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/glass_rule_creator_bitmap_50_0_01_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, prune_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/glass_rule_creator_bitmap_50_005_005_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.05, prune_param_raw=0.05, roulette_selection=True)
# test_all(df, 10, 'results_files/glass_rule_creator_bitmap_50_02_02_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.2, prune_param_raw=0.2, roulette_selection=True)
# test_all(df, 10, 'results_files/glass_rule_creator_bitmap_50_03_03_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.3, prune_param_raw=0.3, roulette_selection=True)
#
# print("BREAST_CANCER")
# df = pd.read_csv('data_files/breast_cancer.csv',
#                  encoding='utf-8', delimiter=';')
# test_all(df, 10, 'results_files/breast_cancer_tree.csv', 1, 5, method=test_tree)
# test_all(df, 10, 'results_files/breast_cancer_random_forest_10trees.csv', 1, 5, method=test_random_forest)
# test_all(df, 10, 'results_files/breast_cancer_random_forest_100trees.csv', 1, 5, method=test_random_forest, forest_trees=100)
# test_all(df, 10, 'results_files/breast_cancer_regression.csv', 1, 5, method=test_regression)
#
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_dict_0_0.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_dict_01_0.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.1)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_dict_0_01.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, prune_param_raw=0.1)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_dict_005_005.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.05, prune_param_raw=0.05)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_dict_02_02.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.2, prune_param_raw=0.2)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_dict_03_03.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.3, prune_param_raw=0.3)
# df = cubes_for_numeric_data(df, 10)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_bitmap_10_0_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, roulette_selection=True)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_bitmap_10_01_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_bitmap_10_0_01_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, prune_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_bitmap_10_005_005_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.05, prune_param_raw=0.05, roulette_selection=True)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_bitmap_10_02_02_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.2, prune_param_raw=0.2, roulette_selection=True)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_bitmap_10_03_03_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.3, prune_param_raw=0.3, roulette_selection=True)
# df = pd.read_csv('data_files/breast_cancer.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,20)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_bitmap_20_0_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, roulette_selection=True)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_bitmap_20_01_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_bitmap_20_0_01_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, prune_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_bitmap_20_005_005_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.05, prune_param_raw=0.05, roulette_selection=True)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_bitmap_20_02_02_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.2, prune_param_raw=0.2, roulette_selection=True)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_bitmap_20_03_03_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.3, prune_param_raw=0.3, roulette_selection=True)
#
# df = pd.read_csv('data_files/breast_cancer.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,50)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_bitmap_50_0_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, roulette_selection=True)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_bitmap_50_01_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_bitmap_50_0_01_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, prune_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_bitmap_50_005_005_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.05, prune_param_raw=0.05, roulette_selection=True)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_bitmap_50_02_02_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.2, prune_param_raw=0.2, roulette_selection=True)
# test_all(df, 10, 'results_files/breast_cancer_rule_creator_bitmap_50_03_03_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.3, prune_param_raw=0.3, roulette_selection=True)
#
#
# print("DIABETIC")
# df = pd.read_csv('data_files/diabetic_data.csv',
#                  encoding='utf-8', delimiter=';')
# test_all(df, 10, 'results_files/diabetic_data_tree.csv', 1, 5, method=test_tree)
# test_all(df, 10, 'results_files/diabetic_data_random_forest_10trees.csv', 1, 5, method=test_random_forest)
# test_all(df, 10, 'results_files/diabetic_data_random_forest_100trees.csv', 1, 5, method=test_random_forest, forest_trees=100)
# test_all(df, 10, 'results_files/diabetic_data_regression.csv', 1, 5, method=test_regression)
#
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_dict_0_0.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_dict_01_0.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.1)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_dict_0_01.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, prune_param_raw=0.1)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_dict_005_005.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.05, prune_param_raw=0.05)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_dict_02_02.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.2, prune_param_raw=0.2)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_dict_03_03.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.3, prune_param_raw=0.3)
# df = cubes_for_numeric_data(df, 10)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_bitmap_10_0_0.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_bitmap_10_01_0.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.1)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_bitmap_10_0_01.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, prune_param_raw=0.1)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_bitmap_10_005_005.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.05, prune_param_raw=0.05)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_bitmap_10_02_02.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.2, prune_param_raw=0.2)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_bitmap_10_03_03.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.3, prune_param_raw=0.3)
# df = pd.read_csv('data_files/diabetic_data.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,20)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_bitmap_20_0_0.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_bitmap_20_01_0.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.1)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_bitmap_20_0_01.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, prune_param_raw=0.1)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_bitmap_20_005_005.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.05, prune_param_raw=0.05)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_bitmap_20_02_02.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.2, prune_param_raw=0.2)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_bitmap_20_03_03.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.3, prune_param_raw=0.3)
#
# df = pd.read_csv('data_files/diabetic_data.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,50)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_bitmap_50_0_0.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_bitmap_50_01_0.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.1)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_bitmap_50_0_01.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, prune_param_raw=0.1)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_bitmap_50_005_005.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.05, prune_param_raw=0.05)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_bitmap_50_02_02.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.2, prune_param_raw=0.2)
# test_all(df, 10, 'results_files/diabetic_data_rule_creator_bitmap_50_03_03.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.3, prune_param_raw=0.3)
# #
# print("INCOME")
# df = pd.read_csv('data_files/income.csv',
#                  encoding='utf-8', delimiter=';')
# test_all(df, 10, 'results_files/income_tree.csv', 1, 5, method=test_tree)
# test_all(df, 10, 'results_files/income_random_forest_10trees.csv', 1, 5, method=test_random_forest)
# test_all(df, 10, 'results_files/income_random_forest_100trees.csv', 1, 5, method=test_random_forest, forest_trees=100)
# test_all(df, 10, 'results_files/income_regression.csv', 1, 5, method=test_regression)
#
# test_all(df, 10, 'results_files/income_rule_creator_dict_0_0.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset)
# test_all(df, 10, 'results_files/income_rule_creator_dict_01_0.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.1)
# test_all(df, 10, 'results_files/income_rule_creator_dict_0_01.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, prune_param_raw=0.1)
# test_all(df, 10, 'results_files/income_rule_creator_dict_005_005.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.05, prune_param_raw=0.05)
# test_all(df, 10, 'results_files/income_rule_creator_dict_02_02.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.2, prune_param_raw=0.2)
# test_all(df, 10, 'results_files/income_rule_creator_dict_03_03.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.3, prune_param_raw=0.3)
# df = cubes_for_numeric_data(df, 10)
# test_all(df, 10, 'results_files/income_rule_creator_bitmap_10_0_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, roulette_selection=True)
# test_all(df, 10, 'results_files/income_rule_creator_bitmap_10_01_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/income_rule_creator_bitmap_10_0_01_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, prune_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/income_rule_creator_bitmap_10_005_005_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.05, prune_param_raw=0.05, roulette_selection=True)
# test_all(df, 10, 'results_files/income_rule_creator_bitmap_10_02_02_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.2, prune_param_raw=0.2, roulette_selection=True)
# test_all(df, 10, 'results_files/income_rule_creator_bitmap_10_03_03_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.3, prune_param_raw=0.3, roulette_selection=True)
# df = pd.read_csv('data_files/income.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,20)
# test_all(df, 10, 'results_files/income_rule_creator_bitmap_20_0_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, roulette_selection=True)
# test_all(df, 10, 'results_files/income_rule_creator_bitmap_20_01_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/income_rule_creator_bitmap_20_0_01_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, prune_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/income_rule_creator_bitmap_20_005_005_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.05, prune_param_raw=0.05, roulette_selection=True)
# test_all(df, 10, 'results_files/income_rule_creator_bitmap_20_02_02_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.2, prune_param_raw=0.2, roulette_selection=True)
# test_all(df, 10, 'results_files/income_rule_creator_bitmap_20_03_03_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.3, prune_param_raw=0.3, roulette_selection=True)
#
# df = pd.read_csv('data_files/income.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,50)
# test_all(df, 10, 'results_files/income_rule_creator_bitmap_50_0_0.csv_R', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, roulette_selection=True)
# test_all(df, 10, 'results_files/income_rule_creator_bitmap_50_01_0.csv_R', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/income_rule_creator_bitmap_50_0_01.csv_R', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, prune_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/income_rule_creator_bitmap_50_005_005.csv_R', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.05, prune_param_raw=0.05, roulette_selection=True)
# test_all(df, 10, 'results_files/income_rule_creator_bitmap_50_02_02.csv_R', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.2, prune_param_raw=0.2, roulette_selection=True)
# test_all(df, 10, 'results_files/income_rule_creator_bitmap_50_03_03.csv_R', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.3, prune_param_raw=0.3, roulette_selection=True)
# #
# print("NBA")
# df = pd.read_csv('data_files/nba_5y.csv',
#                  encoding='utf-8', delimiter=';')
# test_all(df, 10, 'results_files/nba_5y_tree.csv', 1, 5, method=test_tree)
# test_all(df, 10, 'results_files/nba_5y_random_forest_10trees.csv', 1, 5, method=test_random_forest)
# test_all(df, 10, 'results_files/nba_5y_random_forest_100trees.csv', 1, 5, method=test_random_forest, forest_trees=100)
# test_all(df, 10, 'results_files/nba_5y_regression.csv', 1, 5, method=test_regression)
#
# test_all(df, 10, 'results_files/nba_5y_rule_creator_dict_0_0.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_dict_01_0.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.1)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_dict_0_01.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, prune_param_raw=0.1)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_dict_005_005.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.05, prune_param_raw=0.05)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_dict_02_02.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.2, prune_param_raw=0.2)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_dict_03_03.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset, grow_param_raw=0.3, prune_param_raw=0.3)
# df = cubes_for_numeric_data(df, 10)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_bitmap_10_0_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, roulette_selection=True)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_bitmap_10_01_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_bitmap_10_0_01_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, prune_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_bitmap_10_005_005_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.05, prune_param_raw=0.05, roulette_selection=True)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_bitmap_10_02_02_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.2, prune_param_raw=0.2, roulette_selection=True)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_bitmap_10_03_03_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.3, prune_param_raw=0.3, roulette_selection=True)
# df = pd.read_csv('data_files/nba_5y.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,20)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_bitmap_20_0_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, roulette_selection=True)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_bitmap_20_01_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_bitmap_20_0_01_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, prune_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_bitmap_20_005_005_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.05, roulette_selection=True, prune_param_raw=0.05)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_bitmap_20_02_02_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.2, prune_param_raw=0.2, roulette_selection=True)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_bitmap_20_03_03_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.3, prune_param_raw=0.3, roulette_selection=True)
#
# df = pd.read_csv('data_files/nba_5y.csv',
#                  encoding='utf-8', delimiter=';')
# df = cubes_for_numeric_data(df,50)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_bitmap_50_0_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, roulette_selection=True)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_bitmap_50_01_0_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_bitmap_50_0_01_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, prune_param_raw=0.1, roulette_selection=True)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_bitmap_50_005_005_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.05, prune_param_raw=0.05, roulette_selection=True)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_bitmap_50_02_02_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.2, prune_param_raw=0.2, roulette_selection=True)
# test_all(df, 10, 'results_files/nba_5y_rule_creator_bitmap_50_03_03_R.csv', 1, 5, method=test_rule_creator, dataset_type=BitmapDataset, grow_param_raw=0.3, prune_param_raw=0.3, roulette_selection=True)
#
#
# df = pd.read_csv('data_files/mushroom.csv', encoding='utf-8', dalimiter=';')
# test_all(df, 5, 'results_files/mushroom_bitmap_increasing.csv', 15, 5, method=test_rule_creator, dataset_type=BitmapDataset)
#
#
# for i in range(0, 20):
#     df = create_random_dataframe(5, 5000 * (i+1), condition1, 0.1)
#     test_all(df, 5, 'results_files/random_dataset_dict_rows_inc_' + str(i)+'.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset)
#
#
# for i in range(0, 20):
#     df = create_random_dataframe(5 + i * 2, 10000, condition1, 0.1)
#     test_all(df, 5, 'results_files/random_dataset_dict_cols_inc' + str(i) + '.csv', 1, 5, method=test_rule_creator, dataset_type=DictDataset)
#
# for i in range(0, 20):
#     df = create_random_dataframe(5, 5000 * (i+1), condition1, 0.1)
#     test_all(df, 5, 'results_files/random_dataset_bitmap_rows_inc_' + str(i)+'.csv', 1, 5, method=test_rule_creator)
#
# for i in range(0, 20):
#     df = create_random_dataframe(5 + i * 2, 10000, condition1, 0.1)
#     test_all(df, 5, 'results_files/random_dataset_bitmap_cols_inc' + str(i) + '.csv', 1, 5, method=test_rule_creator)


df = pd.read_csv('data_files/audiofeeldb.csv',
                 encoding='utf-8', delimiter=';')
cubes_for_numeric_data(df, 5)
test_all(df, 1, 'results_files/audio.csv', 1, 10, method=test_rule_creator)