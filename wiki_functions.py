import requests
import json
import pandas as pd
from typing import Optional


def get_articles(n: int, lang: str = 'en') -> pd.DataFrame:
    """
    Retrieve sample of n article titles from Wikipedia, using API
    :param n: number of articles
    :param lang: wikipedia language code, from https://meta.wikimedia.org/wiki/Table_of_Wikimedia_projects
    :return: DataFrame with columns "title" and "id" (it's internal wikipedia id of article)
    """
    session = requests.session()
    api_address = f'https://{lang}.wikipedia.org/w/api.php?action=query&list=random&format=json&rnnamespace=0&rnlimit='
    articles = pd.DataFrame(columns=['id', 'title'])
    while len(articles) < n:
        response = session.get(api_address+str(min(500, n - len(articles))))
        batch = json.loads(response.content)['query']['random']
        articles = pd.concat(
            [articles, pd.DataFrame(batch).drop('ns', axis=1)]).drop_duplicates().reset_index(drop=True)
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
    if items_len != 1:
        print(f'Unknown problem with API response, for article {title} {items_len} elements was returned!!')
    return categories
