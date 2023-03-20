from datetime import datetime
from functools import wraps
from time import time
import ast
import pandas as pd
from typing import Optional
from collections.abc import Callable, Sequence


def apply_with_interim_saving(df: pd.DataFrame, f: Callable, col_to_apply: str, new_col: str, csv_name: str,
                              n: int = 1000, verbose: bool = True, one_by_one: bool = True, **kwargs) -> pd.DataFrame:
    """
    Retrieving categories is done one-by-one, and it could take a lot of time. If some connection errors or
    other problems take place, all data could be lost. To prevent this dataframes updating by chunks of 1000 rows,
    and results are saved to csv file after each chunk processing
    :param df: DataFrame (with name of articles)
    :param f: function to apply (get_category)
    :param col_to_apply: self-explaining (column with name of article)
    :param new_col: name of new col, where result of function f will be stored
    :param csv_name: name of backup csv file where results will be saved
    :param n: size of chunk
    :param verbose: get some printed notifications about dataframe processing
    :param one_by_one: if True, use function with apply, if False: give it list of arguments
    and process dict of responses
    :param kwargs: arguments, which will be passed to the f function
    :return: DataFrame with new columns (and side effect: csv file with result is created)
    """
    if new_col not in df.columns:
        df[new_col] = None
        print(f'No column {new_col}. Added')
    old_index_col = 'old_index_temp_column'
    if old_index_col not in df.columns:
        if any(df.index != range(len(df))):
            df[old_index_col] = df.index.copy()
            print(f'Old index moved to {old_index_col}')
    df.reset_index(drop=True, inplace=True)
    if len(df[df[new_col].isna()]) > 0:
        if df[df[new_col].isna()].index.min() > 0:
            print(f'Finish uncompleted calculations...({df.index.max() - df[df[new_col].isna()].index.min()} rows)')
        for pos in range(df[df[new_col].isna()].index.min(), df.index.max(), n):
            if one_by_one:
                df.loc[pos:pos + n, new_col] = df.loc[pos:pos + n, col_to_apply].apply(
                    lambda x: f(x, **kwargs))
            else:
                args_list = df.loc[pos:pos + n, col_to_apply].tolist()
                df.loc[pos:pos + n, new_col] = pd.Series(f(args_list, **kwargs)).reindex(args_list).set_axis(
                    df.loc[pos:pos + n].index)
            df.to_csv(csv_name, index=False)
            if verbose:
                print(pos, datetime.now())
    else:
        print('Calculations complete')
    if old_index_col in df.columns:
        df.index = df[old_index_col].copy()
        df.drop(old_index_col, axis=1, inplace=True)
    return df


def regroup_categories(df: pd.DataFrame, cat_col: str, id_col: str, lists: bool = True,
                       transform_prohibited: bool = False, add_word: Optional[str] = 'Category:') -> pd.DataFrame:
    """
    Transform DataFrame from result of previous stage to ready for next stage (explode and rearranging lists of
    ids of the articles, belongs to each category).
    For example, from [category, list_of_ids_belong_to_category, list_of_infracategories_of_category]
    it will be transformed to
    [infracategory, list_of_ids_belong_to_infracategory]
    :param df: DataFrame, result of previous stage
    :param cat_col: name of column in df, where lists of categories are staged
    :param id_col: name of column in df, where lists of ids of articles are staged
    :param lists: True if ids are in lists and False if they are integers (at first stage)
    :param transform_prohibited: True if you don't want to try any id_col transformation (i.e. if you have titles
    instead of ids)
    :param add_word: word to add to category column
    :return: Transformed DataFrame
    """
    if not transform_prohibited:
        if lists:
            try:
                df[id_col] = df[id_col].apply(ast.literal_eval)
            except ValueError:
                pass
        else:
            df[id_col] = df[id_col].astype(int)
    agg_func = {id_col: 'sum'} if lists else {id_col: pd.Series.tolist}
    df = df.explode(cat_col).groupby(cat_col).agg(agg_func).reset_index()
    df[id_col] = df[id_col].apply(set).apply(list)
    if add_word is not None:
        df[cat_col] = add_word + df[cat_col]
    return df


def data_path(stage: Sequence[str], project: str, data_folder_name: str = 'data') -> str:
    """Make path to csv backups, according to stage name and project name"""
    if len(stage) == 1:
        return f'{data_folder_name}/{project}/{stage[0]}.csv'
    if len(stage) == 2:
        return f'{data_folder_name}/{project}/{stage[0]}_with_{stage[1]}.csv'


def generate_stages(n: int, final_file: str = 'final') -> list[tuple[str, ...], ...]:
    """
    Generate list of stages of calculations. First two stages are always ('title', ) and ('title', 'category') and last
    always will be (<final_file>, )
    :param n: number of infracategories (0 means that there will be only categories). Total amount of stage is n + 3
    :param final_file: name of last stage (and resulting file)
    :return: list of stage tuples
    """
    base = ('title', 'category', 'infra')
    s = [base[:1], base[:-1]]
    if n == 1:
        s = s + [(base[1], base[2] + '1')]
    elif n > 1:
        s = s + [(base[1], base[2] + '1')] + [(f'{base[2]}{i - 1}', f'{base[2]}{i}') for i in range(2, n + 1)]
    return s + [(final_file, )]


def convert_lists(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Convert DataFrame where one column contains lists, but column was read as string and some values are None
    :param df: DataFrame to convert
    :param col: name of column with lists data
    :return: converted column. Data could be resampled: all not-null at first and all null after them
    """
    nonnull_part = df[~df[col].isna()].copy()
    nonnull_part[col] = nonnull_part[col].apply(ast.literal_eval)
    return pd.concat([nonnull_part, df[df[col].isna()]])


def get_args_dict(f: Callable, args: Sequence, kwargs: dict) -> dict:
    """
    By given function and passed args list and kwargs dict, return dictionary with names and values of all arguments,
    passed to the function, including defaults for those which wasn't passed
    :param f: any function
    :param args: all positional arguments, passed to function f
    :param kwargs: all named arguments passed to function f
    :return: dictionary with all parameters of the function. If function have **kwargs unpacking, there will be 'kwargs'
    parameter in resulting dict with all unpacked arguments,
    and also all this arguments will be in resulting dict by themselves
    """
    var_names = f.__code__.co_varnames
    pos_arg_count = min(f.__code__.co_argcount, len(args))
    flags = '{:04b}'.format(f.__code__.co_flags)
    has_args = int(flags[-3])
    has_kwargs = int(flags[-4])
    kwarg_name = [None, var_names[-1]][has_kwargs]
    arg_name = [None, var_names[-1 - has_kwargs]][has_args]
    kwargs.update(zip(var_names[:pos_arg_count], args[:pos_arg_count]))
    if has_kwargs:
        kwargs[kwarg_name] = {i: kwargs[i] for i in kwargs if (i not in var_names) or (i in [arg_name, kwarg_name])}

    if has_args:
        kwargs[arg_name] = args[pos_arg_count:]
        defaults = f.__kwdefaults__ or {}
    else:
        args_from_named = var_names[pos_arg_count: len(args)]
        args_from_named_values = args[pos_arg_count:]
        kwargs.update(zip(args_from_named, args_from_named_values))

        defaults = f.__defaults__ or []
        defaults = dict(zip(
            var_names[-len(defaults) - has_kwargs:(-1 if has_kwargs else None)],
            defaults
        ))
    defaults.update(kwargs)
    return defaults


def timing(printed_args=None):
    """Decorator which measure time of function """
    def decorator(f):
        @wraps(f)
        def wrap(*args, **kwargs):
            ts = time()
            result = f(*args, **kwargs)
            te = time()
            if printed_args is None:
                print(f'function {f.__name__} run by {te - ts:2.4f} sec')
            elif printed_args == 'all':
                print(f'function {f.__name__} with args {args} and kwargs{kwargs} run by {te - ts:2.4f} sec')
            else:
                args_dict = get_args_dict(f, args, kwargs)
                args_want_to_see = {i: args_dict[i] for i in args_dict if i in printed_args}
                print(f'function {f.__name__} with args {args_want_to_see} run by {te - ts:2.4f} sec')
            return result

        return wrap
    return decorator
