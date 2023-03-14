import scipy.sparse
from scipy.sparse import dok_matrix, lil_matrix, spmatrix, csr_matrix
import pandas as pd
from itertools import combinations
from typing import Optional, Callable, Union

from support_functions import timing
from sklearn.metrics import silhouette_score
import numpy as np
from support_functions import regroup_categories, convert_lists, generate_stages, data_path
from sentence_transformers import util


def check_clusterisation(jaccard_matrix, cluster_labels):
    """Check clusterisation quality by Silhouette score, by jaccard distance """
    return silhouette_score(1 - jaccard_matrix, cluster_labels, metric='precomputed')


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
    for _, index_list in df.loc[df.len_ > 2, ids_col].items():
        for i, j in combinations(index_list, 2):
            if max_val is None:
                d[(i, j)] = d.get((i, j), 0) + 1
            else:
                cur_val = d.get((i, j), 0)
                if cur_val < max_val:
                    d[(i, j)] = cur_val + 1
    category_matrix._update(d)
    category_matrix = category_matrix.tocsc()
    category_matrix = (category_matrix + category_matrix.T).tolil()
    return category_matrix


@timing(printed_args=[])
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
    matrix = matrix / (pairwise_sums - matrix + 0.0001)
    return csr_matrix(matrix)


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


def leveled_jakkard_similarity(
        project: str, stages_num: Optional[int] = None, paths: Optional[list[str]] = None, col_names=None, mults=None,
        data_folder_name='data') -> Optional[tuple[pd.DataFrame, spmatrix]]:
    """
    Calculate pairwise Jakkard similarities between articles, using weighted approach: different levels of hierarchy
    have different weights in resulting graph
    :param project: path to folder with files
    :param stages_num: number of files
    :param col_names: name of column with the list of categories if specific files are
    :param mults: weight of each level, if None - all weights =1
    :param paths: paths to specific files with DataFrames with column 'index' (with list of ids) and column with
    list of categories, each of article from 'index' column belongs to
    :param data_folder_name: name of folder with all projects
    :return: dataframe with articles and their 1-st-level categories, sparce matrix with Jakkard similarities
    """
    if paths is None:
        if stages_num is None:
            print('You should set or number of stages or specific paths to files with data!')
            return
        stages = generate_stages(stages_num - 1)[1:-1]
        paths = [data_path(stage, project, data_folder_name) for stage in stages]
        col_names = [stage[1] for stage in stages]
    else:
        if stages_num is not None:
            print('You should set only one parameter - or number of stages or specific paths to files,',
                  'but both were provided')
            return
        if col_names is None:
            print('You should set col_names if you set specific paths to the files')
            return
    if mults is None:
        mults = [1] * len(paths)
    print(f'Loading files {", ".join(paths)}')
    dfs = [pd.read_csv(paths[0]).reset_index()]
    dfs[0] = convert_lists(dfs[0], col_names[0])
    n = len(dfs[0])
    total_cats_per_article = dfs[0].copy()
    total_cats_per_article['cat_count_weighted'] = total_cats_per_article[col_names[0]].apply(len) * mults[0]
    total_cats_per_article.drop(col_names[0], axis=1, inplace=True)
    cats = [filter_categories(regroup_categories(dfs[0], col_names[0], 'index', lists=False), col_names[0])]
    matrix = make_sparce_category_matrix(cats[0], n).asfptype() * mults[0]

    for num, (path, col_name, mult) in enumerate(zip(paths[1:], col_names[1:], mults[1:])):
        df_ = pd.read_csv(path)
        df_ = convert_lists(df_, col_name)
        df_ = convert_lists(df_, 'index')
        df_ = df_[df_[col_names[num]].isin(set(cats[num][col_names[num]]))]
        total_cats_per_article['cat_count_weighted'] += regroup_categories(
            df_, 'index', col_name, lists=True, add_word=None
        ).set_index('index')[col_name].apply(len).reindex(dfs[0].index).fillna(0) * mult
        dfs.append(df_)
        cats.append(regroup_categories(df_, col_name, 'index', lists=True))
        cats[num + 1] = filter_categories(cats[num + 1], col_name)
        matrix += make_sparce_category_matrix(cats[num + 1], n).asfptype() * mult

    matrix = calculate_jakkard(matrix, total_cats_per_article['cat_count_weighted'].values)
    return dfs[0][['title', 'category']], matrix
