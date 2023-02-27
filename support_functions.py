from datetime import datetime
import ast
import pandas as pd


def apply_with_interim_saving(df, f, col_to_apply, new_col, csv_name, n=1000, verbose=True, **kwargs):
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
            df.loc[pos:pos + n, new_col] = df.loc[pos:pos + n, col_to_apply].apply(
                lambda x: f(x, **kwargs))
            df.to_csv(csv_name, index=False)
            if verbose:
                print(pos, datetime.now())
    else:
        print('Calculations complete')
    if old_index_col in df.columns:
        df.index = df[old_index_col].copy()
        df.drop(old_index_col, axis=1, inplace=True)
    return df


def regroup_categories(df, cat_col, id_col, lists=True):
    if lists:
        try:
            df[id_col] = df[id_col].apply(ast.literal_eval)
        except ValueError:
            pass
    agg_func = {id_col: 'sum'} if lists else {id_col: pd.Series.tolist}
    df = df.explode(cat_col).groupby(cat_col).agg(agg_func).reset_index()
    df[id_col] = df[id_col].apply(set).apply(list)
    df[cat_col] = 'Category:' + df[cat_col]
    return df


def data_path(stage, project, data_folder_name='data'):
    if len(stage) == 1:
        return f'{data_folder_name}/{project}/{stage[0]}.csv'
    if len(stage) == 2:
        return f'{data_folder_name}/{project}/{stage[0]}_with_{stage[1]}.csv'


def generate_stages(n):
    base = ('title', 'category', 'infra')
    s = [base[:1], base[:-1]]
    if n == 1:
        s = s + [(base[1], base[2] + '1')]
    elif n > 1:
        s = s + [(base[1], base[2] + '1')] + [(f'{base[2]}{i - 1}', f'{base[2]}{i}') for i in range(2, n + 1)]
    return s + [('final', )]


def convert_lists(df, col):
    nonnull_part = df[~df[col].isna()].copy()
    nonnull_part[col] = nonnull_part[col].apply(ast.literal_eval)
    return pd.concat([nonnull_part, df[df[col].isna()]])
