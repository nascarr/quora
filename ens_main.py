#!/usr/bin/env python
from ensemble import *
from main import *
from ens_model_list import model_dict


def parse_ens_main_args():
    parser = argparse.ArgumentParser(parents=[ens_parser(add_help=False)])
    arg = parser.add_argument
    arg('--main_args', '-a', nargs='+', default=['--mode test -em glove', '--mode test -em wnews', '--mode test -em paragram'], type=str)
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
