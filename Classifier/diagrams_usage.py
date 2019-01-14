
from Classifier.diagrams import produce_diagram_for_one_file_with_errors, cube_diagram

# produce_diagram_for_one_file_with_errors('jakosciowe/random_dataset_rows_inc.csv', 'train_examples', 'time_mean', 'time_std',
#                  'Liczba przykładów', 'Czas w sekundach', 24, 24, 2, tick_size=15, groupby=False)
#
# produce_diagram_for_one_file_with_errors('jakosciowe/features_inc.csv', 'number_of_features', 'time_mean', 'time_std',
#                  'Liczba atrybutów', 'Czas w sekundach', 24, 24, 2, tick_size=15)
#
# produce_diagram_for_one_file_with_errors('all/mushroom_rule_creator_bitmap_inc.csv', 'All train examples', 'Time in seconds', 'Time in seconds',
#                  'Liczba przykładów', 'Czas w sekundach', 24, 24, 2, tick_size=15, groupby=True)
#
#
cube_diagram('uzyte_w_pracy/income_tree.csv',
             'uzyte_w_pracy/income_random_forest_100trees.csv',
             'uzyte_w_pracy/income_regression.csv',
             'uzyte_w_pracy/income_rule_creator_dict_0_0.csv',
             'uzyte_w_pracy/income_rule_creator_bitmap_10_0_0.csv',
             'uzyte_w_pracy/income_rule_creator_bitmap_50_0_0_R.csv',
             'Errors (FP + FN)')


cube_diagram('uzyte_w_pracy/nba_5y_tree.csv',
             'uzyte_w_pracy/nba_5y_random_forest_100trees.csv',
             'uzyte_w_pracy/nba_5y_regression.csv',
             'uzyte_w_pracy/nba_5y_rule_creator_dict_01_0.csv',
             'uzyte_w_pracy/nba_5y_rule_creator_bitmap_10_005_005.csv',
             'uzyte_w_pracy/nba_5y_rule_creator_bitmap_10_005_005_R.csv',
             'Errors (FP + FN)')


cube_diagram('uzyte_w_pracy/phoneme_tree.csv',
             'uzyte_w_pracy/phoneme_random_forest_100trees.csv',
             'uzyte_w_pracy/phoneme_regression.csv',
             'uzyte_w_pracy/phoneme_rule_creator_dict_005_005.csv',
             'uzyte_w_pracy/phoneme_rule_creator_bitmap_10_0_0.csv',
             'uzyte_w_pracy/phoneme_rule_creator_bitmap_10_0_0_R.csv',
             'Errors (FP + FN)')


cube_diagram('uzyte_w_pracy/glass_tree.csv',
             'uzyte_w_pracy/glass_random_forest_100trees.csv',
             'uzyte_w_pracy/glass_regression.csv',
             'uzyte_w_pracy/glass_rule_creator_dict_03_03.csv',
             'uzyte_w_pracy/glass_rule_creator_bitmap_10_02_02.csv',
             'uzyte_w_pracy/glass_rule_creator_bitmap_10_02_02_R.csv',
             'Errors (FP + FN)')


cube_diagram('uzyte_w_pracy/breast_cancer_tree.csv',
             'uzyte_w_pracy/breast_cancer_random_forest_100trees.csv',
             'uzyte_w_pracy/breast_cancer_regression.csv',
             'uzyte_w_pracy/breast_cancer_rule_creator_dict_0_0.csv',
             'uzyte_w_pracy/breast_cancer_rule_creator_bitmap_10_005_005.csv',
             'uzyte_w_pracy/breast_cancer_rule_creator_bitmap_10_0_0_R.csv',
             'Errors (FP + FN)')


cube_diagram('uzyte_w_pracy/mushroom_tree.csv',
             'uzyte_w_pracy/mushroom_random_forest_100trees.csv',
             'uzyte_w_pracy/mushroom_regression.csv',
             'uzyte_w_pracy/mushroom_rule_creator_dict.csv',
             'uzyte_w_pracy/mushroom_rule_creator_bitmap.csv',
             'uzyte_w_pracy/mushroom_rule_creator_bitmap_R.csv',
             'Errors (FP + FN)')