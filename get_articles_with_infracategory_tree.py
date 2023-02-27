import os
from os.path import exists

import requests
import pandas as pd

from wiki_functions import get_category, get_articles
from support_functions import generate_stages, data_path, convert_lists, apply_with_interim_saving, regroup_categories

data_folder_name = 'data'
project = '1k'
number_of_infracategories = 3  # 0 for category only, 1 for categories and categories of categories etc.
articles_num = 1000

stages = generate_stages(number_of_infracategories)
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
                df = convert_lists(df, stage[1])
                print(f'Starting stage {stage_num}. Process {len(df)} rows({stage[0]})...')
            df = apply_with_interim_saving(df, f=get_category, col_to_apply=stage[0], new_col=stage[1],
                                           csv_name=data_path(stage, project), session=requests.session())
        else:
            try:
                print(f'Starting stage {stage_num}. Process {len(df)} rows({stage[0]})...')
            except NameError:
                print(f'Loading data from {data_path(prev_stage, project)}')
                df = pd.read_csv(data_path(prev_stage, project), dtype=str)
                df = convert_lists(df, prev_stage[1])
                df = apply_with_interim_saving(df, f=get_category, col_to_apply=prev_stage[0], new_col=prev_stage[1],
                                               csv_name=data_path(prev_stage, project), session=requests.session())
                print(f'Starting stage {stage_num}. Process {len(df)} rows({stage[0]})...')
            if stage_num == 2:
                df = regroup_categories(df.reset_index(), cat_col=prev_stage[1], id_col='index', lists=False)
            else:
                df = regroup_categories(df, cat_col=prev_stage[1], id_col='index', lists=True)
            if stage_num == len(stages) - 1:
                df.to_csv(path, index=False)
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
            print(f'Stage {stage_num} (get {stage[1]} for {stage[0]}) already processed, check file completeness')
    prev_stage = stage
