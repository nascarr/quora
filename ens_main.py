#!/usr/bin/env python
from ensemble import *
from main import *
from ens_model_list import model_dict


def parse_ens_main_args():
    parser = argparse.ArgumentParser(parents=[ens_parser(add_help=False)])
    arg = parser.add_argument
    arg('--main_args', '-a', nargs='+', default=['--mode test'], type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_ens_main_args()
    main_args = [a.split() for a in args.main_args]
    record_paths = []
    for a in main_args:
        record_path = main(a)
        record_paths.append(record_path)
    ens = Ensemble(record_paths, args.k, model_args='paths')
    ens(args)
