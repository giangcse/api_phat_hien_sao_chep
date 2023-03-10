import os
import dotenv
import json

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# load variables in .env file
dotenv.load_dotenv()
# init elasticsearch connection
es = Elasticsearch(
    hosts=[os.environ['HOST']+':'+os.environ['ES_PORT']],
    http_auth=(os.environ['ELASTIC_USERNAME'], os.environ['ELASTIC_PASSWORD']),
    verify_certs=False
)
# create mapping for data in elasticsearch
if os.path.exists('mapping.json'):
    f = open('mapping.json', encoding='utf8')
    mapping = json.loads(f.read())

    mappings = {
        "properties": {
            mapping['properties']
        }
    }

    es.indices.create(index=os.environ['ES_INDEX'], mappings=mappings)

    import pandas as pd


    bulk_data = []
    for i,row in df.iterrows():
        bulk_data.append(
            {
                "_index": "articles",
                "_source": {        
                    "url": row['Url'],
                    "title": row['Title'],
                    "keywords": row['Keywords'],
                    "content": row['Content'],
                    "author": row['Author'],
                    "len": row['len']
                }
            }
        )
    bulk(es, bulk_data)
else:
    print('mapping.json not found. Create IT!')