# list of possible models for ensemble


model_dict = {
    'wnews': ('Jan_10_2019__21:56:52', '-es 3 -e 10 -em wnews  -hd 150 -we 10 --lrstep 10'), # 827
    'glove': ('Jan_10_2019__22:10:15', '-es 2 -e 9 -hd 150'), # 829
    'paragram': ('Jan_11_2019__11:10:57', '-hd 150 -es 2 -e 10 -em paragram -t lowerspacy -us 0'), #873
    'gnews': ('Jan_11_2019__20:21:01', '-em gnews -es 2 -e 10 -hd 150 -us 0.1'), # 889
    'wnews_test': ('Jan_10_2019__19:33:05_test', '--mode test -em wnews'),
    'glove_test': ('Jan_10_2019__19:34:39_test', '--mode test -em glove'),
}
