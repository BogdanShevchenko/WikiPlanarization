import scipy.sparse
from scipy.sparse import dok_matrix, lil_matrix, spmatrix
import pandas as pd
from itertools import permutations
from typing import Optional, Callable, Union

from support_functions import timing
from sklearn.metrics import silhouette_score
import numpy as np

from support_functions import regroup_categories, convert_lists


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
    print('Total links after clipping:', category_matrix.todense().sum())
    return category_matrix


def calculate_jakkard(matrix: Union[spmatrix, np.matrix], each_node_edges: np.array) -> spmatrix:
    """
    Convert matrix of common categories to jakkard similarity matrix. Used weightened Jakkard as here
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html
    Also, gor Jakkard generalization read
    http://theory.stanford.edu/~sergei/papers/soda10-jaccard.pdf

    :param matrix: affinity matrix (A(i,j) = # of similar categories
    :param each_node_edges: number of categories for each of articles (weighted with same weights as matrix)

    :return: matrix with same size as input matrix
    """
    pairwise_sums = np.add.outer(each_node_edges, each_node_edges)
    matrix = matrix / (pairwise_sums - matrix)
    return matrix


def filter_categories(df, cat_col='category'):
    """Filter categories, which are too wide and doesn't in fact mark some real similarity"""
    exclude = ['[Dd]isambiguation',
               ' stubs$',
               ':Living people',
               '[0-9]+s? (?:births$|deaths)',
               'Category:Deaths (?:by|due|from)',
               '(?:Alcohol-related|Accidental|Road incident|Tuberculosis|Sports?) deaths',
               '(?:century|animal|racehorse|BC) (?:births|deaths)',
               ':Alumni',
               'alumni$',
               'People educated at',
               'Category:People from',
               "people of .+ descent",
               ':[0-9]+(?:th|st)-century.+(?:women|people)$',
               ':Burials at',
               ':[0-9]+ (?:dis)?establishments in',
               'century (?:dis)?establishments', ]
    for token in exclude:
        df = df[~df[cat_col].str.contains(token)]
    return df
