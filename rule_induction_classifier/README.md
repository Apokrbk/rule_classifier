W celu zainstalowania pakietu rule_induction_classifier należy zainstalować wszystkie pakiety wymienione w pliku requirements.txt (np. komendą pip install -r requirements.txt).

Następnie należy zainstalować sam pakiet wywołując komendy w odpowiednim środowisku Python:
- python setup.py build
- python setup.py test (uruchomią się testy jednostkowe)
- python setup.py install (pakiet zostanie zainstalowany)

Po zainstalowaniu pakietu można go używać za pomocą polecenia import. Przykładowo:
- from rule_induction_classifier.rule_creator import RuleCreator
- from rule_induction_classifier.abstract_datasets.bitmap_dataset.bitmap_dataset import BitmapDataset

W klasie RuleCreator dostępne są następujące funkcje:
- __init__(self, dataset_type=BitmapDataset, shuffle_dataset=1, grow_param_raw=0, prune_param_raw=0, roulette_selection=False, split_ratio=2/3) - konstruktor, pierwszy argument odpowiada za sposób przechowywania danych (BitmapDataset - wersja B, DictDataset - wersja A), grow_param_raw i prune_param_raw to parametry służące do dostosowania szczegółowości tworzonych reguł, roulette_selection oznacza użycie selekcji ruletkowej, split_ratio decyduje jaka część zbioru trenującego trafi do zbioru do tworzenia reguł
- fit(self, df_x, df_y) - funkcja odpowiedzialna za utworzenie reguł, df_x to tabela zawierająca informacje o przykładach, df_y to wektor klas odpowiadający przykładom ze zbioru df_x (może przyjmować tylko wartości 0,1)
- predict(self, dataset) -  zwraca wektor klas dla przykładów ze zbioru dataset
- print_rules(self) - wypisuje utworzone reguły
- get_rules(self) - zwraca utworzone reguły