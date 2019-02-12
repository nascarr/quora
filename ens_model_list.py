# list of possible models for ensemble


model_dict = {
    'wnews': ('Jan_10_2019__21:56:52', '-es 3 -e 10 -em wnews  -hd 150 -we 10 --lrstep 10'), # 827
    'glove': ('Jan_10_2019__22:10:15', '-es 2 -e 9 -hd 150'), # 829
    'paragram': ('Jan_11_2019__11:10:57', '-hd 150 -es 2 -e 10 -em paragram -t lowerspacy -us 0'), #873
    'gnews': ('Jan_11_2019__20:21:01', '-em gnews -es 2 -e 10 -hd 150 -us 0.1'), # 889
    'gnews_num': ('Jan_12_2019__22:13:37', '-em gnews -t gnews_num -es 2 -e 10 -us 0.1 -hd 150'), # 927
    'wnews_test': ('Jan_10_2019__19:33:05_test', '--mode test -em wnews'),
    'glove_test': ('Jan_10_2019__19:34:39_test', '--mode test -em glove'),
    'glove_cv': ('Jan_17_2019__17_59_36', '-hd 150 -k 5 -ne 3'),  # 1077
    'paragram_cv': ('Jan_15_2019__04_23_16', '-k 5 -hd 150 -e 10 -em paragram -t lowerspacy -us 0.1'), #983 ! try -ne 3
    'wnews_cv': ('Jan_11_2019__05:09:58', '-es 3 -e 10 -em wnews -m BiLSTMPoolTest -vl -hd 150 -we 10 --lrstep 20 -k 5 -us 0.1'), #855
    'gnews_cv': ('Jan_18_2019__15_06_29', '-em gnews -es 2 -e 10 -hd 150 -us 0.1 -k 5 -ne 3'),  # 1113
    'gnews_num_cv': ('Jan_18_2019__16_45_38', '-em gnews -e 10 -hd 150 -us 0.1 -k 5 -ne 3 -t gnews_num'), # 1119
    'gnews_ph_cv': ('Jan_19_2019__16_05_11', '-em gnews -e 10 -hd 150 -ne 3 -t gnews_ph -k 5 -us 0.1'), #1189
    'gnews_ph_num_cv': ('Jan_19_2019__18_56_02', '-em gnews -e 10 -hd 150 -ne 3 -t gnews_ph_num -k 5 -us 0.1'),  #1195
    'gnews_cv_2': ('Jan_20_2019__08_17_29', '-em gnews -e 12 -hd 150 -ne 3 -k 5 -us 0.1 -lr 0.002'),  #1234
    'wnews_cv_2': ('Jan_21_2019__00_07_54', '-em wnews -e 12 -hd 150 -ne 3 -k 5 -us 0.1 -lr 0.002'), #1288
    'wnews_cv_3': ('Jan_21_2019__22_08_57', '-em wnews -e 12 -hd 150 -ne 3 -k 5 -us 0.1 -lr 0.0025 -we 10'), #1334 ! try --lrstep 2
    'linpool4_cv': ('Feb_01_2019__16_06_28', '-k 5 -m LinPool4 -em glove paragram wnews gnews -we 20 -e 10'), #1501
    'linpool3_cv': ('Feb_03_2019__14_41_27', '-k 5 -m LinPool3 -em glove paragram wnews -e 10 -lr 0.002 -we 20'), #1589
    'linpool3_cv_2': ('Feb_05_2019__12_55_11', '-k 5 -m LinPool3 -em glove paragram wnews -hd 150 -we 20 -lr 0.002 -e 15 -es 3') # 1783
}
