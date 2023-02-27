from wiki_functions import get_articles, get_category
import requests
from datetime import datetime

session = requests.session()
titles = get_articles(100000)
titles.to_csv('titles.csv')

n = 500
t = datetime.now()
for i in range(titles[titles.categories.isna()].index.min(), titles.index.max(), n):
    titles.loc[i:i + n, 'category'] = titles.loc[i: i + n, 'title'].apply(
        lambda x: get_category(x, session=session))
    titles.to_csv('title_with_category.csv', index=False)
    print('\n', i, (datetime.now() - t).total_seconds() / len(titles.loc[i:i + n, 'category']) * 1000)
    t = datetime.now()

