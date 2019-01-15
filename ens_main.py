#!/usr/bin/env python
from ensemble import *
from main import *
from ens_model_list import model_dict


def parse_ens_main_args():
    parser = argparse.ArgumentParser(parents=[ens_parser(add_help=False)])
    arg = parser.add_argument
    arg('--main_args', '-a', nargs='+', default=["-es 3 -e 8 -em wnews -hd 150 -we 10 --lrstep 10 -mv 500000", "-e 5 -hd 150 -mv 1100000", "-e 8 -hd 150 -em paragram -us 0 -mv 850000"], type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_ens_main_args()
    main_args = [a.split() for a in args.main_args]
    record_dirs = []
    for a in main_args:
        record_dir = main(a)
        record_dirs.append(record_dir)
    ens = Ensemble(record_dirs, args.k, model_args='paths')
    ens(args)

# '--mode test -em glove', '--mode test -em wnews', '--mode test -em paragram'
# "-es 3 -e 8 -em wnews -hd 150 -we 10 --lrstep 10 -us 0.1", "-e 5 -hd 150 -us 0.1", "-e 5 -hd 150 -em paragram -us 0"