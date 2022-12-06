import json
import pyodbc
import uvicorn
import tqdm
import urllib3
import os

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from underthesea import sent_tokenize

from fastapi import FastAPI, Form
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union

class SEARCH_BODY(BaseModel):
    query: str
    index_name: int
    method: str
    index: int
    title: Union[str, None] = None
    quantity: Union[int, None] = None


class API:
    def __init__(self) -> None:
        '''
        HÀM KHỞI TẠO
        ------
        Khởi tạo các tham số, các kết nối đến SQL Server, Elasticsearch, tạo API, load model embedding
        '''
        self.TOP_3_LIST = []
        self.BODY_QUERY = {}
        self.CHUNK_SIZE = 16384
        self.CONFIG_FILE = 'config.json'
        # Model embedding
        self.model_embedding = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')
        # Check file config
        if os.path.exists(self.CONFIG_FILE):
            with open(self.CONFIG_FILE, 'r', encoding='utf8') as f:
                config = json.load(f)
                # Init an Elasticsearch connection
                self.es_client = Elasticsearch(
                    'https://' + str(config['elasticsearch']['host']) + ':' + str(config['elasticsearch']['port']),
                    http_auth=(str(config['elasticsearch']['username']), str(config['elasticsearch']['password'])),
                    verify_certs=False
                )
                # Connect to SQL Server and create a cursor
                self.sql_connection = pyodbc.connect(driver='{ODBC Driver 17 for SQL Server}', 
                                                    host=str(config['sql']['host']), database=str(config['sql']['database']),
                                                    user=str(config['sql']['username']), password=str(config['sql']['password']))
                self.cursor = self.sql_connection.cursor()
        # Init an API app
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=['*'],
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*']
        )
        # Create endpoint index
        @self.app.get('/')
        async def index():
            return JSONResponse(status_code=200, content="Kết nối thành công. Vui lòng kiểm tra lại config.")
        # Update database
        @self.app.post('/update')
        async def update():
            '''
            Hàm cập nhật DB từ SQL Server sang Elasticsearch
            -----
            Tìm max_id bên Elasticsearch và so sánh với SQL Server
            '''
            try:
                response = self.es_client.search(
                    index='sangkien_title_description',
                    body={
                        "aggs": {
                            "max_id": { "max": { "field": "id" } }
                        }
                    },
                    ignore=[400]
                )
                max_id = response['aggregations']['max_id']['value']
                result = self.cursor.execute("SELECT SangKienID, TenSangKien, MoTa FROM dbo.TDKT_SangKien WHERE SangKienID > {}".format(max_id))
            except Exception:
                result = self.cursor.execute("SELECT SangKienID, TenSangKien, MoTa FROM dbo.TDKT_SangKien WHERE SangKienID > {}".format(0))

            self.data = []
            for i in result:
                self.data.append({"id": int(i[0]), "tensangkien": str(i[1]), "mota": str(i[2])})
            self.bulk_data()
            return JSONResponse(status_code=200, content="Cập nhật thành công")

        # Set config
        @self.app.post('/set_config')
        async def set_config(request: Request, sql_host: str = Form(...), sql_database: str = Form(...), sql_username: str = Form(...), sql_password: str = Form(...), es_host: str = Form(...), es_port: int = Form(...), es_username: str = Form(...), es_password: str = Form(...)):
            '''
            Hàm tạo/ghi đè file config
            ------
            input:
            - sql_host (str): Host SQL Server
            - sql_database (str): Database SQL Server
            - sql_username (str): Username SQL Server
            - sql_password (str): Password SQL Server
            - es_host (str): Elasticsearch host
            - es_port (int): Elasticsearch port
            - es_username (str): Elasticsearch username
            - es_password (str): Elasticsearch password

            output:
            File JSON: config.json
            '''
            with open(self.CONFIG_FILE, 'w', encoding='utf8') as f:
                json.dump({
                    "sql": {
                        "host": sql_host,
                        "database": sql_database,
                        "username": sql_username,
                        "password": sql_password
                    },
                    "elasticsearch": {
                        "host": es_host,
                        "port": es_port,
                        "username": es_username,
                        "password": es_password
                    }
                }, f)
            f.close()
            return JSONResponse(status_code=200, content="Lưu cài đặt thành công. Vui lòng khởi động lại hệ thống")
        # Create search
        @self.app.post('/search')
        async def search(search_body: SEARCH_BODY):
            self.BODY_QUERY = {"query": search_body.query, "index_name":search_body.index_name, "method": search_body.method, "index": search_body.index, "title": search_body.title}
            return JSONResponse(content=self.find_(search_body.query, search_body.index_name, search_body.method, search_body.index, search_body.title), status_code=200)

        @self.app.post('/check_dao_van')
        async def check_dao_van(search_body: SEARCH_BODY):
            self.BODY_QUERY = {"query": search_body.query, "index_name":search_body.index_name, "method": search_body.method, "index": search_body.index, "title": search_body.title, "quantity": search_body.quantity}
            self.find_best_match_docs(search_body.query, search_body.index_name, search_body.method, search_body.index, search_body.title, search_body.quantity)
            return JSONResponse(content=self.statistic_for_doc(), status_code=200)

    def find_(self, query, index_name, method, index, title):
        '''
        Hàm tìm kiếm
        -----
        input:
        - query (str): Nội dung tìm kiếm
        - index_name (int): Loại nội dung cần tìm: 0 - Tìm theo tên; 1 - Tìm theo mô tả
        - method (str): Thuật toán tìm kiếm: BM25 hoặc simCSE
        - index (int): 0 - Vector hoá cả đoạn mô tả; 1 - Vector hoá từng câu trong mô tả
        output:
        JSON Object
        '''
        search_vector = 'description_vector' if int(index_name)==1 else 'title_vector'
        index = 'sangkien_title_description' if int(index)==0 else 'sangkien_title_each_description'
        if str(method).lower() == 'simcse':
            query_vector = self.embed_text([tokenize(query)])[0]
            if(title):
                script_query = {
                    "script_score": {
                        "query": {
                            "match": {
                                "title": title
                            }
                        }
                        ,
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, '"+search_vector+"') + 0.0",
                            "params": {"query_vector": query_vector}
                        }
                    }
                }
            else:
                script_query = {
                "script_score": {
                    "query": {
                        "match_all": {}
                    }
                    ,
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, '"+search_vector+"') + 0.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        else:
            script_query = {
                "match": {
                "description": {
                    "query": query,
                    "fuzziness": "AUTO"
                }
                }
            }
        response = self.es_client.search(
            index=index,
            body={
                "size": 5,
                "query": script_query,
                "_source": {
                    "includes": ["id", "title", "description"]
                },
            },
            ignore=[400]
        )

        result = []
        for hit in response["hits"]["hits"]:
            result.append({"score": hit["_score"], "title": str(hit["_source"]['title']), "description": str(hit["_source"]['description'])})
        
        return result

    def find_best_match_docs(self, query, index_name, method, index, title, quantity):
        self.TOP_3_LIST = self.find_(query, index_name, method, index, title)[:quantity-1]
        print('***********TOP 3 LIST: {}'.format(self.TOP_3_LIST))
        pass

    def statistic_for_doc(self):
        result = []
        if len(self.TOP_3_LIST) > 0:
            sentences = sent_tokenize(self.BODY_QUERY['query'])
            for item in self.TOP_3_LIST:
                for sentence in sentences:
                    print('+++++++++++++++ Item: {} ------------ Sentence: {}'.format(item, sentence))
                    result.append({"data": self.find_(sentence, self.BODY_QUERY['index_name'], self.BODY_QUERY['method'], 1, item['title']), "query": sentence})

        return (result)


    # Vectorize text
    def embed_text(self, text):
        text_embedding = self.model_embedding.encode(text)
        return text_embedding.tolist()

    # Create an index for Elasticsearch
    def create_index(self, index_name):
        self.es_client.indices.create(
            index=index_name,
            body={
                "settings": {"number_of_shards": 1},
                "mappings": {
                    "properties": {
                        "order": { 
                            "type": "text",
                            "fielddata": True
                            },
                        "title_vector": 
                            {
                                "type": "dense_vector",
                                "dims": 768
                            },
                        "title": {"type": "text"},
                        "description": {"type": "text"},
                        "description_vector": 
                            {
                                "type": "dense_vector",
                                "dims": 768
                            }
                    }
                },
            },
            ignore=400,
        )

    def generate_actions(self, index):
        if int(index)==0:
            for row in self.data:
                title = tokenize(row["tensangkien"])
                title_vector = self.embed_text(title)

                description = tokenize(row["mota"])
                description_vector = self.embed_text(description)

                doc = {
                    "id": int(row["id"]),
                    "title": row["tensangkien"],
                    "title_vector": title_vector,
                    "description": row["mota"],
                    "description_vector": description_vector
                }
                yield doc
        else:
            for row in self.data:
                title = tokenize(row["tensangkien"])
                title_vector = self.embed_text(title)

                sentences = sent_tokenize(row["mota"])

                for sentence in sentences:
                    description = tokenize(sentence)
                    description_vector = self.embed_text(description)

                    doc = {
                        "id": int(row["id"]),
                        "title": row["tensangkien"],
                        "title_vector": title_vector,
                        "description": sentence,
                        "description_vector": description_vector
                    }
                    yield doc


    def bulk_data(self):
        index_name = ['sangkien_title_description', 'sangkien_title_each_description']
        for i in index_name:
            self.create_index(i)
            for ok, action in streaming_bulk(
                client=self.es_client, index=i, actions=self.generate_actions(0 if i=='sangkien_title_description' else 1),
            ):
                print(ok)

api = API()

if __name__=='__main__':
    config = uvicorn.Config("api:api.app", host='0.0.0.0', port=5000, reload="True")
    server = uvicorn.Server(config)
    server.run()