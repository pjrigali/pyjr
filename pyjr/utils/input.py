from dataclasses import dataclass
from typing import Optional
from pyjr.utils.base import _to_list, _prep, _max, _min, _sum


def set_values(input: dict, default: dict, default_dic: dict):
    arg_dic = {'data': None, 'value_type': None, 'na_handling': None, 'std_value': None, 'median_value': None,
               'cap_zero': None, 'ddof': None}
    input_keys_dic = {i: True for i in _to_list(data=input.keys())}
    default_key_lst = _to_list(data=arg_dic.keys())
    for key in default_key_lst:
        if key in input_keys_dic:
            arg_dic[key] = input[key]
        else:
            arg_dic[key] = default[key]
            default_dic[key] = default[key]
    return arg_dic['data'], arg_dic['value_type'], arg_dic['na_handling'], arg_dic['std_value'], \
           arg_dic['median_value'], arg_dic['cap_zero'], arg_dic['ddof']

def _build_dics(args):
    temp_input, temp_default = args
    temp_input = {key: temp_input[key] for key in temp_default.keys()}
    inputs = {key: temp_input[key] for key, val in temp_default.items() if temp_input[key] != temp_default[key]}
    return inputs, temp_default


@dataclass
class Args:

    def __init__(self, inputs: Optional[dict] = None, default: Optional[dict] = None, args = None):

        # Check if args passed
        if args is not None:
            inputs, default = _build_dics(args=args)

        # Initial methods
        self._input = inputs
        self._default_dic = {}
        self._data, self._value_type, self._na_handling, self._std_value, self._median_value, self._cap_zero, \
        self._ddof = set_values(input=inputs, default=default, default_dic=self._default_dic)

        # Build New Data
        self._clean_data = _prep(data=self._data, value_type=self._value_type, na_handling=self._na_handling,
                                 std_value=self._std_value, median_value=self._median_value, cap_zero=self._cap_zero,
                                 ddof=self._ddof)

    def __repr__(self):
        return 'ArgsClass'

    @property
    def data(self) -> list:
        return self._clean_data

    @property
    def input_data(self):
        return self._data

    @property
    def value_type(self) -> str:
        return self._value_type

    @property
    def na_handling(self) -> str:
        return self._na_handling

    @property
    def std_value(self) -> str:
        return self._std_value

    @property
    def median_value(self) -> str:
        return self._median_value

    @property
    def cap_zero(self) -> str:
        return self._cap_zero

    @property
    def ddof(self) -> str:
        return self._ddof

    @property
    def default(self) -> dict:
        return self._default_dic

    @property
    def input(self) -> dict:
        return self._input

    @property
    def min(self):
        return _min(self._clean_data)

    @property
    def max(self):
        return _max(self._clean_data)

    @property
    def len(self) -> int:
        return len(self._clean_data)

    @property
    def sum(self):
        return _sum(self._clean_data)