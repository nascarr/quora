#!/usr/bin/env python

import os


pyfiles = ['create_test_datasets', 'utils', 'dataloader', 'preprocess', 'models', 'learner', 'main']
pydir = '..'
kaggle_script = 'kaggle_script.py'


def read_files(files):
    paths = [os.path.join(pydir, (f + '.py')) for f in files]
    content = []
    for p in paths:
        with open(p, 'r') as f:
            content.append(f.readlines())
    return content


def split_content(content):
    imports = []
    code = []
    for c in content:
        for line in c:
            if line.find('import') >= 0:
                imports.append(line)
            else:
                code.append(line)
    return imports, code


def check_import(im):
    # check if import is necessary
    check = True
    for f in pyfiles:
        if im.find(f) >= 0:
            return False
    return check


def make_kaggle_script():
    content = read_files(pyfiles)
    imports, code = split_content(content)

    with open(kaggle_script, 'w') as kaggle:
        kaggle.write('#!/usr/bin/env python\n')
        for im in imports:
            if check_import(im):
                kaggle.write(im)
        for c in code:
            kaggle.write(c)


if __name__ == '__main__':
    make_kaggle_script()
