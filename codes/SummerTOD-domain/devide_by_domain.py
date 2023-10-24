import json
import os
import argparse


def main(cfg):

    woz_dir = './data/MultiWOZ_{}'.format(cfg.version)

    dial_by_domain = os.path.join(woz_dir, 'dial_by_domain.json')

    with open(dial_by_domain, 'r') as fin:
        lines = json.loads(fin.read().lower())

    test_domain = set()
    train_domain = set()

    for line in lines:
        domains = line.split('-')
        is_test = True
        for domain in domains:
            if domain not in cfg.target_domains:
                is_test = False
                break
        if is_test:
            print('test', line)
            for fn in lines[line]:
                test_domain.add(fn)
            continue
        is_train = True
        for domain in domains:
            if domain in cfg.target_domains:
                is_train = False
                break
        if is_train:
            print('train', line)
            for fn in lines[line]:
                train_domain.add(fn)

    dev_list_path = os.path.join(woz_dir, 'valListFile.json') if cfg.version == '2.0' else os.path.join(woz_dir, 'valListFile.txt')
    with open(dev_list_path, 'r', encoding="utf-8") as fin:
        fns = fin.read()
        fns = fns.lower()
        fns = fns.splitlines()
    test_list_path = os.path.join(woz_dir, 'testListFile.json') if cfg.version == '2.0' else os.path.join(woz_dir, 'testListFile.txt')
    with open(test_list_path, 'r', encoding="utf-8") as fin:
        ff = fin.read()
        ff = ff.lower()
        fns += ff.splitlines()
    
    non_train_list = set()
    for fn in fns:
        non_train_list.add(fn)
    train_list = []
    for fn in train_domain:
        if fn not in non_train_list:
            train_list.append(fn)

    test_list = list(test_domain)

    with open(os.path.join(woz_dir, 'trainListFileALL.json'), 'w') as fout:
        for fn in train_list:
            fout.write(fn + '\n')

    with open(os.path.join(woz_dir, 'testListFileALL.json'), 'w') as fout:
        for fn in test_list:
            fout.write(fn + '\n')
    
    with open(os.path.join(woz_dir, 'valListFileALL.json'), 'w') as fout:
        for fn in test_list:
            fout.write(fn + '\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-target_domains", type=str, default='hospital-police')
    parser.add_argument("-version", type=str, default="2.1",
                       choices=["2.0", "2.1", "2.2"])

    cfg = parser.parse_args()
    cfg.target_domains = cfg.target_domains.split('-')

    print(cfg.target_domains)

    main(cfg)
