from dataclasses import dataclass
from typing import Union, List, Tuple
import pandas as pd
import numpy as np
import random
from pyjr.utils.data import Data
from pyjr.utils.preprocess import PreProcess
from pyjr.utils.base import _unique_values, _check_list, _sum, _round_to


def _check_names(name, name_list) -> bool:
    name_dic = {name: True for name in name_list}
    if name not in name_dic:
        return True
    else:
        raise AttributeError("{} already included in names list".format(name))


def _check_len(len1, len2) -> bool:
    if len1 == len2:
        return True
    else:
        raise AttributeError("(len1: {} ,len2: {} ) Lengths are not the same.".format((len1, len2)))


def _add_column(arr1, arr2) -> np.ndarray:
    new_arr = np.ones((arr1.shape[0], arr1.shape[1] + 1))
    for i in range(arr1.shape[1]):
        new_arr[:, i] = arr1[:, i]
    new_arr[:, new_arr.shape[1] - 1] = arr2[:, 0]
    return new_arr


class ModelingData:

    __slots__ = ["x_data", "x_data_names", "len", "y_data", "y_data_name", "x_train", "x_test", "x_valid", "y_train",
                 "y_test", "y_valid", "train_ind", "test_ind", "valid_ind"]

    def __init__(self):
        self.x_data = None
        self.x_data_names = []
        self.len = None
        self.y_data = None
        self.y_data_name = None
        self.x_train = None
        self.x_test = None
        self.x_valid = None
        self.y_train = None
        self.y_test = None
        self.y_valid = None
        self.train_ind = None
        self.test_ind = None
        self.valid_ind = None

    def __repr__(self):
        return "ModelingData"

    def add_xdata(self, data: Union[Data, PreProcess]):
        if self.x_data is None:
            if isinstance(data, Data):
                self.x_data = data.array(axis=1)
            else:
                self.x_data = np.array(data.data).reshape(data.len, 1)
            self.len = data.len
            self.x_data_names.append(data.name)
        else:
            if _check_len(len1=self.len, len2=data.len) is True and _check_names(name=data.name, name_list=self.x_data_names) is True:
                self.x_data_names.append(data.name)
                if isinstance(data, Data):
                    self.x_data = _add_column(arr1=self.x_data, arr2=data.array(axis=1))
                else:
                    self.x_data = _add_column(arr1=self.x_data, arr2=np.array(data.data).reshape(data.len, 1))
        return self

    def add_ydata(self, data: Union[Data, PreProcess]):
        if _check_len(len1=self.len, len2=data.len):
            if isinstance(data, Data):
                self.y_data = data.array(axis=1)
            else:
                self.y_data = np.array(data.data).reshape(data.len, 1)
            self.y_data_name = data.name
            return self

    def train_test_validate(self, train: Union[int, float] = 0.70, test: Union[int, float] = 0.20,
                            valid: Union[int, float] = 0.10):

        if isinstance(train, float):
            train = int(np.floor(train * float(self.len)))
        if isinstance(test, float):
            test = int(np.floor(test * float(self.len)))
        if isinstance(valid, float):
            valid = int(np.floor(valid * float(self.len)))

        if train + test + valid > self.len:
            raise AttributeError("Train, test, valid add up to more than the length of the data.")

        vals = tuple(range(self.len))
        self.train_ind = random.sample(vals, train)
        train_dic = {i: True for i in self.train_ind}
        self.x_train = self.x_data[self.train_ind]
        self.y_train = self.y_data[self.train_ind]

        vals = [val for val in vals if val not in train_dic]
        self.test_ind = random.sample(vals, test)
        test_dic = {i: True for i in self.test_ind}
        self.x_test = self.x_data[self.test_ind]
        self.y_test = self.y_data[self.test_ind]

        if valid != 0:
            self.valid_ind = [val for val in vals if val not in test_dic]
            self.x_valid = self.x_data[self.valid_ind]
            self.y_valid = self.y_data[self.valid_ind]

        return self.x_train, self.y_train, self.x_test, self.y_test, self.x_valid, self.y_valid

    def add_multiple_xdata(self, data: Union[List[Union[Data, PreProcess]], Tuple[Union[Data, PreProcess]]]):
        """Calls the add_xdata method to add multiple Data's"""
        for i in data:
            self.add_xdata(data=i)
        return self

    def get_balance(self):
        """Returns the data balance for Y_data between train, test, and valid"""
        train_lst = _check_list(data=self.y_train.reshape(1, self.y_train.shape[0])[0])
        test_lst = _check_list(data=self.y_test.reshape(1, self.y_test.shape[0])[0])
        dic = {"train": _unique_values(data=train_lst, count=True),
               "test": _unique_values(data=test_lst, count=True)}

        if self.y_valid is not None:
            valid_lst = _check_list(data=self.y_valid.reshape(1, self.y_valid.shape[0])[0])
            dic["valid"] = _unique_values(data=valid_lst, count=True)

        final_dic = {i: {} for i in dic.keys()}
        for key, val in dic.items():
            for key1, val1 in val.items():
                final_dic[key][key1] = _round_to(data=val[key1] / _sum(data=_check_list(data=val.values())),
                                                 val=100, remainder=True)
        return final_dic
