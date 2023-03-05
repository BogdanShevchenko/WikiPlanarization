import scipy.sparse
from scipy.sparse import dok_matrix, lil_matrix, spmatrix
import pandas as pd
from itertools import permutations
from typing import Optional

from support_functions import timing


@timing(printed_args=['n'])
def make_sparce_category_matrix(df: pd.DataFrame, n: int, ids_col: str = 'index',
                                max_val: Optional[int] = None) -> lil_matrix:
    """
    Convert DataFrame with categories and ids to scipy sparce matrix.
    :param df: DataFrame with column with categories (or infracategories) and column with list of ids, which belong to
    each category
    :param n: total amount of ids
    :param ids_col: name of column with ids
    :param max_val: set if you want not to count common categories if there are more than max_val of them
    :return: sparce matrix n X n where value in cell (i, j) is amount of common categories of article i and article j
    """
    df['len_'] = df[ids_col].apply(len)
    print('Added', df.eval('len_* (len_ - 1) / 2').sum(), 'new edges')
    category_matrix = dok_matrix((n, n), dtype=int)
    df[ids_col] = df[ids_col].apply(sorted).apply(tuple)
    d = {i: 1 for i in df.loc[df.len_ == 2, ids_col].values}
    category_matrix._update(d)
    d = {}
    for _, index_list in df.loc[df.len_ > 2, ids_col].items():
        for i, j in permutations(index_list, 2):
            if i < j:
                if max_val is None:
                    d[(i, j)] = d.get((i, j), 0) + 1
                else:
                    cur_val = d.get((i, j), 0)
                    if cur_val < max_val:
                        d[(i, j)] = cur_val + 1
    category_matrix._update(d)
    category_matrix = (category_matrix + category_matrix.T).tolil()
    category_matrix.setdiag(0)
    return category_matrix