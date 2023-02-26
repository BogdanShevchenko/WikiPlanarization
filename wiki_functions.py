import requests
import json
import pandas as pd


def get_articles(n, lang='en', verbose=False):
    session = requests.session()
    api_address = f'https://{lang}.wikipedia.org/w/api.php?action=query&list=random&format=json&rnnamespace=0&rnlimit='
    articles = pd.DataFrame(columns=['id', 'title'])
    while len(articles) < n:
        if verbose:
            print(len(articles))
        response = session.get(api_address+str(min(500, n - len(articles))))
        batch = json.loads(response.content)['query']['random']
        articles = pd.concat(
            [articles, pd.DataFrame(batch).drop('ns', axis=1)]).drop_duplicates().reset_index(drop=True)
    return articles


def get_category(title, lang='en', session=None):
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
    if items_len != 1:
        print(f'Unknown problem with API response, for article {title} {items_len} elements was returned!!')
    return categories
