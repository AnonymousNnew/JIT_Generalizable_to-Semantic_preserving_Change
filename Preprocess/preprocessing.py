import csv
from copy import copy
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from variable import features_check_before_pre_process


class Preprocessing:

    def __init__(self, NAME_PROJECT, name_file, p):
        self.df = pd.read_csv(os.path.join(NAME_PROJECT, name_file))
        self.path = os.path.join(NAME_PROJECT, name_file)
        self.Y = None
        self.X = None
        self.name_project = NAME_PROJECT
        self.project = p

    def main(self, train_test_split=True):
        self.preprocessing()
        self.Y = self.df.pop('commit insert bug?')
        self.X = self.df
        if train_test_split:
            raise Exception()
        else:
            self.X['commit insert bug?'] = self.Y
            self.X.to_csv(self.name_project + f"/all_after_preprocess.csv")

    def preprocessing(self):

        self.df.rename(columns={'blame commit': 'commit insert bug?'}, inplace=True)

        del self.df['added_lines+removed_lines']
        del self.df['added_lines-removed_lines']

        self.df = self.df.loc[(self.df['mode'] == 'M') | (self.df['mode'] == 'A')]

        features_to_drop = ['adhoc', 'MATH-']  # parent # 'current',
        self.df = self.df.drop(
            columns=list(filter(lambda c: any(map(lambda f: f in c, features_to_drop)), self.df.columns)), axis=1)

        self.df = self.df[features_check_before_pre_process + ['commit insert bug?'] + ['commit', 'file_name']]

