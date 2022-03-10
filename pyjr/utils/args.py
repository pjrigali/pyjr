from base import _check_list, _replace_na, _to_type
from base import _remove_nan, _mean, _std, _percentile


def native_sum_args(args) -> Union[float, int]:
    """

    Calculate Sum of a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :return: Returns the Sum.
    :rtype: float
    :example: *None*
    :note: *None*

    """
    # data = _prep(data=data, value_type=value_type, na_handling=na_handling, std_value=std_value,
    #              median_value=median_value, cap_zero=cap_zero, ddof=ddof)
    with _prep_args(args=args) as data:
        with len(data) as lst_len:
            if lst_len > 1:
                return sum(data)
            elif lst_len == 0:
                return 0.0
            else:
                return data


def native_mean_args(args) -> Union[float, int]:
    """

    Calculate Mean of a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :return: Returns the mean.
    :rtype: float
    :example: *None*
    :note: *None*

    """
    data, value_type, na_handling, std_value, median_value, cap_zero, ddof = args
    with _prep_args(args=args) as data:
        with len(data) as lst_len:
            if lst_len != 0:
                return _to_type(value=sum(data) / lst_len, value_type=value_type)
            else:
                return _to_type(value=0.0, value_type=value_type)


def native_variance_args(args) -> Union[float, int]:
    """

    Calculate Variance of a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :param ddof: Set the degrees of freedom, default is 1. *Optional*
    :type ddof: int
    :return: Returns the Variance.
    :rtype: float
    :example: *None*
    :note: *None*

    """
    data, value_type, na_handling, std_value, median_value, cap_zero, ddof = args
    with _prep_args(args=args) as data:
        with native_mean_args(args=args) as mu:
            return _to_type(value=sum((x - mu) ** 2 for x in data) / (len(data) - ddof), value_type=value_type)


def native_std_args(args) -> Union[float, int]:
    """

    Calculate Standard Deviation of a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :param ddof: Set the degrees of freedom, default is 1. *Optional*
    :type ddof: int
    :return: Returns the Standard Deviation.
    :rtype: float
    :example: *None*
    :note: *None*

    """
    return _to_type(data=native_variance_args(args=args) ** .5, value_type=value_type)


def native_median_args(args) -> Union[float, int]:
    """

    Calculate Median of a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :return: Returns the Median.
    :rtype: float
    :example: *None*
    :note: If multiple values have the same count, will return the mean.
        Median is used if there is an odd number of same count values.

    """
    data, value_type, na_handling, std_value, median_value, cap_zero, ddof = args
    with _prep_args(args=args) as data:
        with sorted(data) as sorted_lst:
            with len(data) as lst_len:
                with (lst_len - 1) // 2 as index:
                    if lst_len % 2:
                        return _to_type(value=sorted_lst[index], value_type=value_type)
                    else:
                        new_data = [sorted_lst[index]] + [sorted_lst[index + 1]]
                        new_args = (new_data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
                        return native_mean_args(args=new_args)


def _replacement_value_args(args) -> float:
    """
    Calculate desired replacement for Nan values.

    :param data: Input data.
    :type data: list.
    :param na_handling: Desired Nan value handling method. {zero, mu, std, median}
    :type na_handling: str.
    :param std_val: Desired Standard Deviation to use.
    :type std_val: int.
    :param cap_zero: Whether to cap the value at zero.
    :type cap_zero: bool.
    :param median_val: Desired percentile to use.
    :type median_val: float.
    :return: Replacement value.
    :note: If mean - 3 * std is less than 0, may confuse results.
    """
    data, value_type, na_handling, std_value, median_value, cap_zero, ddof = args
    if na_handling == 'zero':
        return 0.0
    elif na_handling == 'mu':
        return _mean(data=_remove_nan(data=data))
    elif na_handling == 'std':
        data_new = _remove_nan(data=data)
        val = _mean(data=data_new) - (_std(data=data_new, ddof=ddof) * std_value)
        if cap_zero:
            if val > 0:
                return val
            else:
                return 0
        else:
            return val
    elif na_handling == 'median':
        return _percentile(data=data, length=len(data), q=median_value, val_type='float')
    elif na_handling == 'none':
        return None


def _prep_args(args):
    data, value_type, na_handling, std_value, median_value, cap_zero, ddof = args
    with _check_type(data=_check_list(data=data), value_type=value_type) as data_lst:
        new_args = (data_lst, value_type, na_handling, std_value, median_value, cap_zero, ddof)
        return _replace_na(data=data_lst, replacement_value=_replacement_value_args(args=new_args))