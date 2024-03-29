import scipy.sparse
from scipy.sparse import dok_matrix, lil_matrix, spmatrix, csr_matrix
import pandas as pd
from itertools import combinations
from typing import Optional, Callable, Union, Sequence

from support_functions import timing
from sklearn.metrics import silhouette_score
import numpy as np
from support_functions import regroup_categories, convert_lists, generate_stages, data_path, fillna_list
from sentence_transformers import util


def affinity_to_dist(affinity_matrix: spmatrix) -> np.matrix:
    """Change Jaccard similarity matrix to Jaccard distance matrix"""
    dist_matrix = 1 - affinity_matrix.todense().A
    np.fill_diagonal(dist_matrix, 0)
    return dist_matrix


def check_clusterisation(jaccard_dist: np.matrix, cluster_labels: Sequence) -> float:
    """Check clusterisation quality by Silhouette score, by jaccard distance (which is 1-jaccard similarity)"""
    return silhouette_score(jaccard_dist, cluster_labels, metric='precomputed')


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
    print(f'Adding {df.eval("len_ * (len_ - 1) / 2").sum()} new edges')
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
def calculate_jaccard(matrix: Union[spmatrix, np.matrix], each_node_edges: np.array, parts: int=5000) -> spmatrix:
    """
    Convert matrix of common categories to jaccard similarity matrix. Used weightened Jaccard as here
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html
    Also, gor Jaccard generalization read
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
               "(?:^|:)Living people",
               '[0-9]+s? (?:births$|deaths)',
               '(?:^|:)Deaths (?:by|due|from)',
               '(?:Alcohol-related|Accidental|Road incident|Tuberculosis|Sports?) deaths',
               '(?:century|animal|racehorse|BC) (?:births|deaths)',
               '(?:^|:)Alumni',
               'alumni$',
               'People educated at',
               '(?:^|:)People from',
               "people of .+ descent",
               '(?:^|:)[0-9]+(?:th|st)-century.+(?:women|people)$',
               '(?:^|:)Burials at',
               '(?:^|:)[0-9]+ (?:dis)?establishments in',
               'century (?:dis)?establishments',
               '(?:^|:)Years',
               'by time',
               'Main topic classifications',
               '(?:^|:)Humans$',
               'by decade',
               'Individual apes',
               ' beginnings$',
               'by time',
               'by country',
               'by type and year$',
               'by nationality',
               'by year',
               'Categories by',
               'by continent',
               'Works by',
               '(?:^|:)[0-9][0-9][0-9][0-9]s?$',
               'by century',
               'by occupation',
               'Populated places by',
               '(?:p|P)eople by university',
               'by religion'
               ]
    for token in exclude:
        df = df[~df[cat_col].astype(str).str.contains(token)]
    return df


def leveled_jaccard_similarity(
        project: str, stages_num: Optional[int] = None, paths: Optional[list[str]] = None,
        col_names: Optional[list[str]] = None, mults: Optional[list[int]] = None,
        data_folder_name='data') -> Optional[tuple[pd.DataFrame, spmatrix]]:
    """
    Calculate pairwise Jaccard similarities between articles, using weighted approach: different levels of hierarchy
    have different weights in resulting graph
    :param project: path to folder with files
    :param stages_num: number of files
    :param col_names: name of column with the list of categories if specific files are
    :param mults: weight of each level, if None - all weights =1
    :param paths: paths to specific files with DataFrames with column 'index' (with list of ids) and column with
    list of categories, each of article from 'index' column belongs to
    :param data_folder_name: name of folder with all projects
    :return: dataframe with articles and their 1-st-level categories, sparce matrix with Jaccard similarities
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
        mults = np.ones(len(paths))
    else:
        #  multiply multipliers by some coefficient, so that they will be close to ints. It will not change results
        #  because we use Jaccard measure (it is not changing if all weights are increased simultaneously)
        #  this is done for memory optimisation
        mults = np.array(mults)
        mults = mults / mults[mults != 0].min()
        while np.abs((np.rint(mults) - mults)[mults != 0] / mults[mults != 0]).max() > 0.01:
            mults = mults * 10
        mults = np.rint(mults).astype(int)

    print(f'Loading files {", ".join(paths)}')
    df0 = pd.read_csv(paths[0]).reset_index()
    df0 = convert_lists(df0, col_names[0])
    no_disambig_cond = ~df0.set_index('title').category.astype(str).str.contains('isambig')
    n = len(df0)
    total_cats_per_article = df0.copy()
    total_cats_per_article['cat_count_weighted'] = total_cats_per_article[col_names[0]].apply(len) * mults[0]
    total_cats_per_article.drop(col_names[0], axis=1, inplace=True)
    cats = filter_categories(regroup_categories(df0, col_names[0], 'index', lists=False), col_names[0])
    matrix = make_sparce_category_matrix(cats, n) * mults[0]

    for num, (path, col_name, mult) in enumerate(zip(paths[1:], col_names[1:], mults[1:])):
        df_ = pd.read_csv(path)
        df_ = convert_lists(df_, col_name)
        df_ = convert_lists(df_, 'index')
        df_ = df_[df_[col_names[num]].isin(set(cats[col_names[num]]))]
        total_cats_per_article['cat_count_weighted'] += regroup_categories(
            df_, 'index', col_name, lists=True, add_word=None
        ).set_index('index')[col_name].apply(len).reindex(df0.index).fillna(0) * mult
        cats = regroup_categories(df_, col_name, 'index', lists=True)
        cats = filter_categories(cats, col_name)
        matrix += make_sparce_category_matrix(cats, n) * mult

    matrix = matrix.astype(np.float32)
    matrix = calculate_jaccard(matrix, total_cats_per_article['cat_count_weighted'].values.astype(np.float32))
    df0 = filter_categories(df0.explode(col_names[0]), col_names[0]).groupby('title').agg(
        {col_names[0]: pd.Series.tolist}
    ).reindex(df0['title'])
    df0[col_names[0]] = fillna_list(df0[col_names[0]], [])
    df0 = df0.loc[no_disambig_cond]
    matrix = matrix[no_disambig_cond][:, no_disambig_cond]
    return df0.reset_index(), matrix
