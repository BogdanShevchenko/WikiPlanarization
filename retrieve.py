import json
from typing import Optional
import os
from os.path import exists

import requests
import pandas as pd

from support_functions import generate_stages, data_path, convert_lists, apply_with_interim_saving, regroup_categories


def get_articles(n: int, lang: str = 'en', id_: bool = False) -> pd.DataFrame:
    """
    Retrieve sample of n article titles from Wikipedia, using API
    :param n: number of articles
    :param lang: wikipedia language code, from https://meta.wikimedia.org/wiki/Table_of_Wikimedia_projects
    :param id_: add column with internal wikipedia id
    :return: DataFrame with columns "title" and "id" (if id_ is True)
    """
    session = requests.session()
    api_address = f'https://{lang}.wikipedia.org/w/api.php?action=query&list=random&format=json&rnnamespace=0&rnlimit='
    cols = ['id', 'title'] if id_ else ['title']
    articles = pd.DataFrame(columns=cols)
    while len(articles) < n:
        response = session.get(api_address+str(min(500, n - len(articles))))  # maximum 500 article per request
        batch = json.loads(response.content)['query']['random']
        articles = pd.concat(
            [articles, pd.DataFrame(batch)[cols]]).drop_duplicates().reset_index(drop=True)
    return articles


def get_category(title: str, lang: str = 'en', session: Optional[requests.Session] = None) -> list[str]:
    """
    Retrieve categories of article by its caption using API. Only unhidden categories.
    :param title: title of article, as it shown on page
    :param lang: wikipedia language code, from https://meta.wikimedia.org/wiki/Table_of_Wikimedia_projects
    :param session: use requests.Session() for massive retrieving
    :return: list of categories (without "Category:" prefix)
    """
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "categories",
        "titles": title,
        "clshow": "!hidden"
    }
    if session is None:
        session = requests.session()

    data = session.get(url=url, params=params).json()["query"]["pages"]
    items_len = len(data.items())
    try:
        categories = [cat['title'].replace('Category:', '') for k, v in data.items() for cat in v['categories']]
    except KeyError:
        print(title, 'no categories')
        categories = []
    if items_len != 1:  # Never get this error but who knows
        print(f'Unknown problem with API response, for article {title} {items_len} elements was returned!!')
    return categories


def get_articles_with_infracategories(articles_num, number_of_infracategories, project,
                                      data_folder_name='data', final_stage='final'):
    """
    Get articles title, categories for each title, categories of their categories (infracategories) and so on.
    Data will be saved in folder <data_folder_name>/<project> in multiple csv files with next structure
    one file will be "title"|"list of categories of article"
    and other
    "category"|"list of article ids in this category"|"categories which this category belongs"
    and also there will be one file named "final.csv", where all data will be grouped
    structure of fi
    ids are number of row in first file
    :param articles_num: how many articles do you want to have
    :param number_of_infracategories: 0 means that there will be only categories,
    1 for categories and categories of categories etc.
    :param project: name of subfolder in data folder, where all files will be saved
    :param data_folder_name: name of data folder (just "data" by default)
    :param final_stage: name of final stage and it (plus .csv) will be the name of resulting file
    :return: None
    """
    stages = generate_stages(number_of_infracategories, final_stage)
    if not exists(data_folder_name):
        os.mkdir(data_folder_name)
    if not exists(f'{data_folder_name}/{project}'):
        os.mkdir(f'{data_folder_name}/{project}')
    prev_stage = None
    for stage_num, stage in enumerate(stages):
        path = data_path(stage, project)
        if not exists(path):
            if stage_num == 0:
                print('Starting stage 0...')
                df = get_articles(articles_num)
                df.to_csv(path, index=False)
            elif stage_num == 1:
                try:
                    print(f'Starting stage {stage_num}. Process {len(df)} rows({stage[0]})...')
                except NameError:
                    df = pd.read_csv(data_path(prev_stage, project), dtype=str)
                    print(f'Starting stage {stage_num}. Process {len(df)} rows({stage[0]})...')
                df = df.reset_index()
                df = df[['title', 'index']]
                df = apply_with_interim_saving(df, f=get_category, col_to_apply=stage[0], new_col=stage[1],
                                               csv_name=data_path(stage, project), session=requests.session())

            else:
                try:
                    print(f'Starting stage {stage_num}. Process {len(df)} rows({stage[0]})...')
                except NameError:
                    print(f'Loading data from {data_path(prev_stage, project)}')
                    df = pd.read_csv(data_path(prev_stage, project), dtype=str)
                    df = convert_lists(df, prev_stage[1])
                    df = apply_with_interim_saving(
                        df, f=get_category, col_to_apply=prev_stage[0], new_col=prev_stage[1],
                        csv_name=data_path(prev_stage, project), session=requests.session()
                    )
                    print(f'Starting stage {stage_num}. Process {len(df)} rows({stage[0]})...')
                df = regroup_categories(df, cat_col=prev_stage[1], id_col='index', lists=stage_num != 2)
                if stage_num == len(stages) - 1:
                    res = []
                    for stage_ in stages[1:-1]:
                        df_ = pd.read_csv(data_path(stage_, project), dtype=str)
                        df_ = convert_lists(df_, stage_[1])
                        df_ = regroup_categories(df_.reset_index(), cat_col=stage_[1], id_col='index',
                                                 lists=stage_ != stages[1])
                        df_.columns = ['obj', 'ids']
                        df_['obj_name'] = stage_[1]
                        res.append(df_)
                    pd.concat(res).to_csv(path, index=False)
                    print('Final stage complete')
                    break
                df = apply_with_interim_saving(df, f=get_category, col_to_apply=stage[0], new_col=stage[1],
                                               csv_name=data_path(stage, project), session=requests.session())
        else:
            if stage_num > 1:
                print('Stage complete')
            if stage_num == 0:
                print(f'Stage {stage_num} (get random articles) already processed, skip')
            elif stage_num == len(stages) - 1:
                print('Final stage complete')
            else:
                print(f'Stage {stage_num} (get {stage[1]} for {stage[0]}) already start processed, check file completeness')
        prev_stage = stage
