#!/usr/bin/env python

import os
import glob


main_scripts = ['learner.py', 'main.py', 'ensemble.py', 'ens_main.py']
exclude_scripts = ['models_dev.py', 'stop_words.py', 'correlation.py', 'read_csv.py']
pydir = '.'
kaggle_script = 'kaggle/kaggle_script.py'


def read_files(files):
    paths = [os.path.join(pydir, f) for f in files]
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


def check_import(im, pyfiles):
    # check if import is necessary
    check = True
    modules = [pf[:-3] for pf in pyfiles]
    for m in modules:
        if im.find(m) >= 0:
            return False
    return check


def make_kaggle_script():
    pyfiles = list(set(glob.glob('*.py')) - set(main_scripts) - set(exclude_scripts))
    pyfiles.extend(main_scripts)
    content = read_files(pyfiles)
    imports, code = split_content(content)

    with open(kaggle_script, 'w') as kaggle:
        kaggle.write('#!/usr/bin/env python\n')
        for im in imports:
            if check_import(im, pyfiles):
                kaggle.write(im)
        for c in code:
            kaggle.write(c)


if __name__ == '__main__':
    make_kaggle_script()
