import pickle
import sys
import os
import re
import pathlib

import csv

import git
import numpy as np
from operator import itemgetter
from datetime import datetime
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
from pydriller import Repository
from nltk import FreqDist
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, StratifiedKFold


def write_file_func(file, commit, k_fold):
    before = file.source_code_before
    after = file.source_code
    try:
        if after is not None:
            with open(f"Files/{NAME_PROJECT}/{str(k_fold)}/{commit.hash}_after_{file.filename}", 'w') as f:
                f.write(after)
        if before is not None:
            with open(f"Files/{NAME_PROJECT}/{str(k_fold)}/{commit.hash}_before_{file.filename}", 'w') as f:
                f.write(before)
        else:  # add file
            with open(f"Files/{NAME_PROJECT}/{str(k_fold)}/{commit.hash}_before_{file.filename}", 'w') as f:
                f.write("")
    except Exception as e:
        print(e)
        pass


def parser_commit_message(list_commit_message):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    tokens = []
    new_list_msg = []
    msg_before = ""
    for msg in list_commit_message:
        array_words = word_tokenize(msg)
        array_words = [stemmer.stem(plural).lower() for plural in array_words]
        array_words = [w for w in array_words if not w in stop_words]
        new_list_msg.append(array_words)
        if not msg_before.__eq__(msg):
            tokens.extend(array_words)
        msg_before = msg
    # return new_list_msg
    tokenizer = Tokenizer(filters=" ")
    tokenizer.fit_on_texts(tokens)
    # return [" ".join(i) for i in new_list_msg], tokenizer.word_index
    return rare_word(tokens, new_list_msg), tokenizer.word_index


def rare_word(tokens, list_data):
    freq_dist = FreqDist(tokens)
    tokens_ = {key: val for key, val in freq_dist.items() if val > 3}
    value = []
    for msg in list_data:
        new_msg = []
        for let in msg:
            if isinstance(let, list):
                val_1 = []
                for item in let:
                    if item in tokens_.keys():
                        val_1.append(item)
                new_msg.append(" ".join(val_1))
            else:
                if let in tokens_.keys():
                    new_msg.append(let)
        if isinstance(let, list):
            value.append(new_msg)
        else:
            value.append(" ".join(new_msg))
    return value


def parser_commit_change(list_code_change):
    def fix_first_deleted(list_code_change):
        new_list_code_change = []
        for i in list_code_change:
            new_val = []
            first = True
            for val in i:
                if val.startswith("<deleted>") and first:
                    split_ = val.split(";")
                    new_val.append(split_[0])
                    for j in range(1, len(split_)):
                        new_val.append("<deleted> " + split_[j])
                    first = False
                else:
                    new_val.append(val)
            new_list_code_change.append(new_val)
        return new_list_code_change
    list_code_change = fix_first_deleted(list_code_change)

    list_code_change_ = []
    tokens = []
    for change in list_code_change:
        line_ = []
        for line in change:
            new_line = re.sub(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', '<NUM>', line)
            new_line = [new_line.split(">")[0] + ">"] + word_tokenize(" ".join(re.split("../WriteFileCommit", new_line.split(">")[1])))
            # line_.append(" ".join(new_line))
            line_.append(new_line)
            tokens.extend(new_line)
        list_code_change_.append(line_)
    tokenizer = Tokenizer(filters=" ")
    tokenizer.fit_on_texts(tokens)
    # return list_code_change_, tokenizer.word_index
    return rare_word(tokens, list_code_change_), tokenizer.word_index


def extract_commit_data(file, commit):
    file_added = ["<added> " + i[1].strip() for i in file.diff_parsed['added'] if not check_comment(i[1].strip())]
    file_deleted = ["<deleted>" + i[1].strip() for i in file.diff_parsed['deleted'] if not check_comment(i[1].strip())]
    commit_msg = commit.msg
    return commit_msg, file_added + file_deleted


def read_commit_blame():
    """
    Reading the file that contains the instances (commit + file) that induced defect.
    This file Written in call the SZZ algorithm.

    :return: dataframe contain two columns: commit and file name
    :rtype: DataFrame
    .. notes:: You must run the function main_szz() in file issues_extractor.py before execute this function
    """
    dataset = pd.read_csv(
        str(pathlib.Path(
            __file__).parent.resolve()) + rf"/../GAN_VS_RF/Data/{NAME_PROJECT}/blame/pydriller_{NAME_PROJECT}_bugfixes_bic.csv",
        delimiter=',')
    dataset['filename'] = dataset['filename'].str.replace("\\", r'/')
    dataset['filename'] = dataset['filename'].str.split("/").str[-1]
    del dataset['bugfix_commit']
    return dataset.drop_duplicates()


def check_comment(i):
    return i.startswith("*") or i == "\n" or i.startswith("/*") or \
           i.startswith("//") or i.startswith("*/") or i == "" or i == ""


def extract_data_transformation(added, delete, real_commit):
    commit_msg = repo.commit(real_commit).message
    file_added = ["<added> " + i.strip() for i in added if not check_comment(i.strip())]
    file_deleted = ["<deleted>" + i.strip() for i in delete if not check_comment(i.strip())]
    return commit_msg, file_added + file_deleted


def get_transformation_name(k):
    return [x[0].split("/")[-1] for x in os.walk('/sise/home/shir0/JavaTransformer/diff_after_transformation/' + NAME_PROJECT + "//" + str(k) + "//")]


def get_modification_transformation_data():
    for k in range(1, NUMBER_FOLD + 1):
        name_transformation = get_transformation_name(k)
        set_commit = []
        for dir_transformation in name_transformation:
            index = 0
            list_commit_msg, list_code_change, ids, labels = [], [], [], []
            if dir_transformation == NAME_PROJECT or dir_transformation == "":
                continue
            if not os.path.exists(f'/sise/home/shir0/JavaTransformer/diff_after_transformation/{NAME_PROJECT}/{str(k)}/{dir_transformation}/transform.csv'):
                dir = [x[2] for x in os.walk(f'/sise/home/shir0/JavaTransformer/diff_after_transformation/{NAME_PROJECT}/{str(k)}/{dir_transformation}/')][0]
                transform_dir = [i for i in dir if i != NAME_PROJECT and i != ""]
                merge = []
                for i in transform_dir:
                    merge.append(pd.read_csv(f'/sise/home/shir0/JavaTransformer/diff_after_transformation/{NAME_PROJECT}/{str(k)}/{dir_transformation}/{i}', engine ='python', delimiter="!!!!!#$$$", header=None))
                pd.concat(merge).to_csv(f'/sise/home/shir0/JavaTransformer/diff_after_transformation/{NAME_PROJECT}/{str(k)}/{dir_transformation}/transform.csv', header=None, index=False)
            data = pd.read_csv(os.path.join(f'/sise/home/shir0/JavaTransformer/diff_after_transformation/{NAME_PROJECT}/{str(k)}/{dir_transformation}/transform.csv'),
                               delimiter="!!!!!#$$$", header=None)
            for index_, row in data.iterrows():
                split_c_file = row[0].split('!@#:')
                real_commit = split_c_file[0].replace('"', "")
                real_file = split_c_file[1]
                split_added = split_c_file[2].split('!@#+')
                added = split_added[: -1]
                deleted = split_added[-1].split('!@#-')
                commit_msg, code_change = extract_data_transformation(added, deleted, real_commit)
                if f"{real_commit}_{real_file}" not in set_commit:
                    set_commit.append(f"{real_commit}_{real_file}")
                ids.append(f"{real_commit}_{real_file}")
                list_commit_msg.append(commit_msg)
                list_code_change.append(code_change)
                value = blame_dataset[
                    (blame_dataset['filename'] == real_file + ".java") & (blame_dataset['bic'] == real_commit)]
                if value.empty:
                    labels.append(0)
                else:
                    labels.append(1)
                index += 1
                if index % 100 == 0:
                    print(f"index_{index}")
            list_code_change, dic_codes = parser_commit_change(list_code_change)
            list_commit_msg, dic_msg = parser_commit_message(list_commit_msg)
            print(f"len(labels) {len(labels)}")
            if not os.path.exists(f"Data/{NAME_PROJECT}/{str(k)}/{dir_transformation}/"):
                os.makedirs(f'Data/{NAME_PROJECT}/{str(k)}/{dir_transformation}/')
            with open(f'Data/{NAME_PROJECT}/{str(k)}/{dir_transformation}/{NAME_PROJECT}_transformation.pkl',
                      'wb') as f:
                pickle.dump((list(np.array(ids)), list(np.array(labels)), list(np.array(list_commit_msg)),
                             list(np.array(list_code_change))), f)


def get_modification_real_project(write_file=True, ids_test=None, k_fold=1):
    ids, list_commit_msg, list_code_change, labels, methods = [], [], [], [], {}
    end_with = ".java"
    index = 0
    for commit in Repository(URL_REAL_REPO, only_modifications_with_file_types=[end_with]).traverse_commits():
        commit_mod = filter(
            lambda x: x.filename.endswith(end_with) and not x.filename.lower().endswith("test" + end_with),
            commit.modified_files)
        for file in commit_mod:
            if file.change_type.name == 'ADD' or file.change_type.name == 'DELETE':
                continue
            try:  # Continue if there is a crash
                if write_file and f"{commit.hash}_{file.filename}" in ids_test:
                    write_file_func(file, commit, k_fold)
                else:
                    commit_msg, code_change = extract_commit_data(file, commit)
                    if not code_change:
                        continue
                    value = blame_dataset[
                        (blame_dataset['filename'] == file.filename) & (blame_dataset['bic'] == commit.hash)]
                    if value.empty:
                        labels.append(0)
                    else:
                        labels.append(1)
                    methods[f"{commit.hash}_{file.filename}"] = [method.name for method in
                                                                 file.changed_methods]

                    ids.append(f"{commit.hash}_{file.filename}")
                    list_commit_msg.append(commit_msg)
                    list_code_change.append(code_change)
            except Exception as e:
                print(e)
                continue

            index += 1
            if index % 100 == 0:
                print(f"index_{index}")
    if not write_file:
        list_code_change, dic_codes = parser_commit_change(list_code_change)
        list_commit_msg, dic_msg = parser_commit_message(list_commit_msg)
        write_real_project(ids, list_code_change, list_commit_msg, dic_msg, dic_codes, methods, labels)


def write_real_project(ids, list_code_change, list_commit_msg, dic_msg, dic_codes, methods, labels):
    skf = StratifiedKFold(n_splits=NUMBER_FOLD, shuffle=True, random_state=12)
    index_k = 1
    for ids_train, ids_test in skf.split(ids, labels):
        if not os.path.exists(f"Data/{NAME_PROJECT}/{str(index_k)}"):
            os.mkdir(f"Data/{NAME_PROJECT}/{str(index_k)}")
        if not os.path.exists(f"Files/{NAME_PROJECT}/{str(index_k)}"):
            os.mkdir(f"Files/{NAME_PROJECT}/{str(index_k)}")
        with open(f'Data/{NAME_PROJECT}/{str(index_k)}/{NAME_PROJECT}_train.pkl', 'wb') as f:
            pickle.dump((list(np.array(ids)[ids_train]), list(np.array(labels)[ids_train]),
                         list(np.array(list_commit_msg)[ids_train]),
                         list(np.array(list_code_change)[ids_train])), f)
        with open(f'Data/{NAME_PROJECT}/{str(index_k)}/{NAME_PROJECT}_test.pkl', 'wb') as f:
            pickle.dump(
                (list(np.array(ids)[ids_test]), list(np.array(labels)[ids_test]),
                 list(np.array(list_commit_msg)[ids_test]),
                 list(np.array(list_code_change)[ids_test])), f)
        with open(f'Data/{NAME_PROJECT}/{str(index_k)}/{NAME_PROJECT}_dict.pkl', 'wb') as f:
            pickle.dump((dic_msg, dic_codes), f)
        with open(f'Files/{NAME_PROJECT}/{str(index_k)}/{NAME_PROJECT}_methods.csv', 'w') as f:
            for key in methods.keys():
                f.write("%s," % key)
                for i in methods[key]:
                    f.write("%s," % i)
                f.write("\n")
        get_modification_real_project(write_file=True, ids_test=read_ids_test(index_k), k_fold=index_k)
        index_k += 1

    # X_train, X_test, y_train, y_test = train_test_split(ids, labels, test_size=0.2, random_state=42, stratify=labels)
    # ids_train = list(np.where(np.isin(ids, X_train))[0])
    # ids_test = list(np.where(np.isin(ids, X_test))[0])
    #
    # with open(f'Data/{NAME_PROJECT}/{NAME_PROJECT}_train.pkl', 'wb') as f:
    #     pickle.dump((list(np.array(ids)[ids_train]), list(np.array(labels)[ids_train]),
    #                  list(np.array(list_commit_msg)[ids_train]),
    #                  list(np.array(list_code_change)[ids_train])), f)
    # with open(f'Data/{NAME_PROJECT}/{NAME_PROJECT}_test.pkl', 'wb') as f:
    #     pickle.dump(
    #         (list(np.array(ids)[ids_test]), list(np.array(labels)[ids_test]), list(np.array(list_commit_msg)[ids_test]),
    #          list(np.array(list_code_change)[ids_test])), f)
    # with open(f'Data/{NAME_PROJECT}/{NAME_PROJECT}_dict.pkl', 'wb') as f:
    #     pickle.dump((dic_msg, dic_codes), f)
    # with open(f'Files/{NAME_PROJECT}/{NAME_PROJECT}_methods.csv', 'w') as f:
    #     for key in methods.keys():
    #         f.write("%s," % key)
    #         for i in methods[key]:
    #             f.write("%s," % i)
    #         f.write("\n")


def read_ids_test(number_fold=1):
    data = pickle.load(open(os.path.join("../WriteFileCommit/Data", NAME_PROJECT, str(number_fold), f"{NAME_PROJECT}_test.pkl"), 'rb'))
    ids, labels, msgs, codes = data
    return ids


def create_dir():
    if not os.path.exists(f"Files/{NAME_PROJECT}"):
        os.mkdir(f"Files/{NAME_PROJECT}")
    if not os.path.exists(f"Data/{NAME_PROJECT}"):
        os.mkdir(f"Data/{NAME_PROJECT}")
    if not os.path.exists(f"Transformations/{NAME_PROJECT}"):
        os.mkdir(f"Transformations/{NAME_PROJECT}")


if __name__ == '__main__':
    # This project have two tasks:
    # 1. (real_project = True) Write JAVA file (before and after) from test data set
    # 2. (real_project = True) Extract features of real project and transformation (commit message and code changes)
    # NAME_PROJECT = 'zeppelin'  # 'zookeeper'
    # URL_REAL_REPO = 'https://github.com/apache/zeppelin'  # 'https://github.com/apache/zookeeper'
    real_project = bool(int(sys.argv[1]))
    NUMBER_FOLD = 5
    # name_projects = ['tika', 'jspwiki', 'openwebbeans', 'zookeeper']
    # url_projects = ['https://github.com/apache/tika', 'https://github.com/apache/jspwiki',
    #                 'https://github.com/apache/openwebbeans','https://github.com/apache/zookeeper']

    # name_projects = ['xmlgraphics-batik']
    # url_projects = ['https://github.com/apache/xmlgraphics-batik']

    name_projects = ['commons-lang']
    url_projects = ['https://github.com/apache/commons-lang']

    # name_projects = ['commons-lang', 'tapestry-5', 'knox', 'xmlgraphics-batik', 'deltaspike']
    # url_projects = ['https://github.com/apache/commons-lang', 'https://github.com/apache/tapestry-5',
    #                 'https://github.com/apache/knox', 'https://github.com/apache/xmlgraphics-batik',
    #                 'https://github.com/apache/deltaspike']

    # name_projects = ['zeppelin']
    # url_projects = ['https://github.com/apache/zeppelin']

    # name_projects = ['manifoldcf', 'openwebbeans', 'zookeeper']  # 'zeppelin'
    # url_projects = ['https://github.com/apache/manifoldcf',
    #                 'https://github.com/apache/openwebbeans', 'https://github.com/apache/zookeeper']  # https://github.com/apache/zeppelin

    # name_projects = ['kafka']
    # url_projects = ['https://github.com/apache/kafka']
    # 'kafka', 'https://github.com/apache/kafka'
    # name_projects = ['zeppelin', 'commons-collections', 'jspwiki', ]
    # url_projects = ['https://github.com/apache/zeppelin', 'https://github.com/apache/commons-collections',
    #                 'https://github.com/apache/jspwiki']
    for NAME_PROJECT, URL_REAL_REPO in zip(name_projects, url_projects):
        REPO_PATH = '../Repo/' + NAME_PROJECT
        blame_dataset = read_commit_blame()
        repo = git.Repo(REPO_PATH)
        create_dir()
        if real_project:
            get_modification_real_project(write_file=False)
        else:
            get_modification_transformation_data()
