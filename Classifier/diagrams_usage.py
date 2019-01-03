
from Classifier.diagrams import produce_diagram_for_one_file_with_errors, cube_diagram

produce_diagram_for_one_file_with_errors('jakosciowe/mushroom_rule_creator_bitmap_inc.csv', 'All train examples', 'Time in seconds', 'Time in seconds',
                 'Liczba przykładów', 'Czas w sekundach', 24, 24, 2, tick_size=15, groupby=True)

produce_diagram_for_one_file_with_errors('jakosciowe/ALL_KUR_CZASY_features.csv', 'number_of_features', 'time_mean', 'time_std',
                 'Liczba przykładów', 'Czas w sekundach', 24, 24, 2, tick_size=15)


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