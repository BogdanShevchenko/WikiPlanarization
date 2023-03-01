Project to make a "map" of Wikipedia, based on categories of sample of articles

For collecting data use:
```python
from retrieve import get_articles_with_infracategories
data_folder_name = 'data'
project = '1k'
number_of_infracategories = 3  # 0 for category only, 
# 1 for categories and categories of categories etc.
articles_num = 1000
get_articles_with_infracategories(articles_num, number_of_infracategories, 
                                  project, data_folder_name='data')
```

Wikipedia API is not very fast (and they could block you, if you will use treads 
and make too many requests per second), it takes near 3 minutes for 1000 requests. So, if you
want to have big dataset and a lot of infracategories, it will take many hours to get 
all data.

You can try to download data from
* https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-category.sql.gz
* https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-categorylinks.sql.gz

It will be much faster, but there are hidden categories in dump and names could be not
consistent

Anyway, if you do, lib `mwsql` will help you to read dumps