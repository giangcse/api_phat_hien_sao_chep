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

# Model search body json
class SEARCH_BODY(BaseModel):
    query: str
    index_name: int
    method: str
    index: int
    title: Union[str, None] = None
    quantity: Union[int, None] = None
# Model config body json
class CONFIG(BaseModel):
    sql_host: str
    sql_database: str
    sql_username: str
    sql_password: str
    es_host: str
    es_port: int = 1553
    es_username: str = 'elastic'
    es_password: str
    similarity_threshold: float

class UPDATE(BaseModel):
    id: int

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
        self.config = {}
        # Model embedding
        self.model_embedding = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')
        # Check file config
        print('*********** start init *********** ')
        config_file_path = os.path.join('./web/',self.CONFIG_FILE)
        if os.path.exists(config_file_path):
            print('*********** CONFIG_FILE existed *********** ')
            with open(config_file_path, 'r', encoding='utf8') as f:
                self.config = json.load(f)
                print('*********** CONFIG_FILE loaded *********** ')
                # Init an Elasticsearch connection
                self.es_client = Elasticsearch(
                    'https://' + str(self.config['elasticsearch']['host']) + ':' + str(self.config['elasticsearch']['port']),
                    http_auth=(str(self.config['elasticsearch']['username']), str(self.config['elasticsearch']['password'])),
                    verify_certs=False
                )
                print('*********** Elasticsearch inited *********** ')
                # Connect to SQL Server and create a cursor
                self.sql_connection = pyodbc.connect(driver='{ODBC Driver 17 for SQL Server}', 
                                                    host=str(self.config['sql']['host']), database=str(self.config['sql']['database']),
                                                    user=str(self.config['sql']['username']), password=str(self.config['sql']['password']))
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
        async def update(update: UPDATE):
            '''
            Hàm cập nhật DB từ SQL Server sang Elasticsearch
            -----
            Tìm max_id bên Elasticsearch và so sánh với SQL Server
            '''
            try:
                indices = ['sangkien_title_description', 'sangkien_title_each_description']
                response = self.es_client.delete_by_query(
                    index=indices,
                    body={
                        "query":{
                            "bool": {
                                "must": {
                                    "match": {"id": int(update.id)}
                                }
                            }
                        }
                    },
                    ignore=[400]
                )

                result = self.cursor.execute("SELECT SangKienID, TenSangKien, MoTa FROM dbo.TDKT_SangKien WHERE SangKienID = {}".format(update.id))
            except Exception:
                result = self.cursor.execute("SELECT SangKienID, TenSangKien, MoTa FROM dbo.TDKT_SangKien WHERE SangKienID > {}".format(0))

            self.data = []
            for i in result:
                self.data.append({"id": int(i[0]), "tensangkien": str(i[1]), "mota": str(i[2])})
            self.bulk_data()
            return JSONResponse(status_code=200, content="Cập nhật thành công")

        # Set config
        @self.app.post('/set_config')
        async def set_config(config_body: CONFIG):
            '''
            Hàm tạo/ghi đè file config
            ------
            input (JSON Object):
            - sql_host (str): Host SQL Server
            - sql_database (str): Database SQL Server
            - sql_username (str): Username SQL Server
            - sql_password (str): Password SQL Server
            - es_host (str): Elasticsearch host
            - es_port (int): Elasticsearch port
            - es_username (str): Elasticsearch username
            - es_password (str): Elasticsearch password
            - similarity_threshold (Int): Ngưỡng xác định 2 câu giống nhau
            output:
            File JSON: config.json
            '''
            with open(self.CONFIG_FILE, 'w', encoding='utf8') as f:
                json.dump({
                    "sql": {
                        "host": config_body.sql_host,
                        "database": config_body.sql_database,
                        "username": config_body.sql_username,
                        "password": config_body.sql_password
                    },
                    "elasticsearch": {
                        "host": config_body.es_host,
                        "port": config_body.es_port,
                        "username": config_body.es_username,
                        "password": config_body.es_password
                    },
                    "similarity_threshold": config_body.similarity_threshold
                }, f)
            f.close()
            return JSONResponse(status_code=200, content="Lưu cài đặt thành công. Vui lòng khởi động lại hệ thống")
        # Create search
        @self.app.post('/search')
        async def search(search_body: SEARCH_BODY):
            self.BODY_QUERY = {"query": search_body.query, "index_name":search_body.index_name, "method": search_body.method, "index": search_body.index, "title": search_body.title}
            content , _ = self.find_(search_body.query, search_body.index_name, search_body.method, search_body.index, search_body.title)
            return JSONResponse(content , status_code=200)

        @self.app.post('/check_dao_van')
        async def check_dao_van(search_body: SEARCH_BODY):
            self.BODY_QUERY = {"query": search_body.query, "index_name":search_body.index_name, "method": search_body.method, "index": search_body.index, "title": search_body.title, "quantity": search_body.quantity}
            self.find_best_match_docs(search_body.query, search_body.index_name, search_body.method, search_body.index, search_body.title, search_body.quantity)
            return JSONResponse(content=self.statistic_for_doc(), status_code=200)

    def find_(self, query, index_name, method, index, id):
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
            if(id):
                script_query = {
                    "script_score": {
                            "query": {
                                "bool": {
                                    "match": {
                                                "id": id
                                            }
                                }
                            },
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
                "query": script_query,
                "_source": {
                    "includes": ["id", "title", "description"]
                },
            },
            ignore=[400]
        )
        
        result = []
        if "hits" in response:
            for hit in response["hits"]["hits"]:
                result.append({"score": hit["_score"] * 100, 
                                "id": hit["_source"]['id'],
                                "title": str(hit["_source"]['title']), 
                                "description": str(hit["_source"]['description'])})
        
        return result , response["hits"]['total']['value']

    def find_best_match_docs(self, query, index_name, method, index, id, quantity):
        self.TOP_3_LIST , _ = self.find_(query, index_name, method, index, id)[:quantity-1]
        print('***********TOP 3 LIST: {}'.format(self.TOP_3_LIST))
        pass

    def statistic_for_doc(self):
        resultList = []
        compareDict = {}
        similar_sentences = []
        if len(self.TOP_3_LIST) > 0:
            sentences = sent_tokenize(self.BODY_QUERY['query'])
            for item in self.TOP_3_LIST:
                for sentence in sentences:
                    print('+++++++++++++++ Item: {} ------------ Sentence: {}'.format(item, sentence))
                    result , total_sentences = self.find_(sentence, self.BODY_QUERY['index_name'], self.BODY_QUERY['method'], 1, item['id'])
                    for item in result:
                        if (float(item['score']) > self.config['similarity_threshold']):
                            compareDict['title'] = item['title']
                            similar_sentences.append(item)

                compareDict['number_similar'] = len(similar_sentences)
                compareDict['total_sentences'] = total_sentences
                try:
                    compareDict['percentage_similarity'] = str(round((len(similar_sentences) / total_sentences * 100), 2))
                except ZeroDivisionError:
                    compareDict['percentage_similarity'] = str(0)
                compareDict['similar_sentences'] = similar_sentences
                if compareDict['number_similar'] > 0:
                    resultList.append(compareDict)
                    compareDict = {}
                    similar_sentences = []

        return (resultList)


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