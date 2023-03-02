from scipy.sparse import  dok_matrix
import pandas as pd
from support_functions import regroup_categories, convert_lists
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import jaccard
import numpy as np
from datetime import datetime
from itertools import permutations
from scipy.sparse import csgraph
from sklearn.cluster import SpectralClustering
import spacy
from itertools import product
import networkx as nx


def make_sparce_category_matrix(df, n, cat_col, ids_col):
    """
    Convert DataFrame with categories and ids to scipy sparce matrix.
    :param df: DatFrame with columns 
    :param n:
    :param cat_col:
    :param ids_col:
    :return:
    """
    df['len_'] = df['index'].apply(len)
    category_matrix = dok_matrix((n, n), dtype=int)
    df['index'] = df['index'].apply(tuple)
    d = {i: 1 for i in df.loc[df.len_ == 2, 'index'].values}
    category_matrix._update(d)
    d = {}
    for _, index_list in df.loc[df.len_ > 2, 'index'].items():
        for i, j in permutations(index_list, 2):
            if i > j:
                d[(i, j)] = d.get((i, j), 0) + 1
    category_matrix._update(d)
    category_matrix = (category_matrix + category_matrix.T).tolil()
    category_matrix.setdiag(0)
    return category_matrix