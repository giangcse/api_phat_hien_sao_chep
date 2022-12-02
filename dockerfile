FROM laudio/pyodbc:1.0.38

WORKDIR /source

# Add your source files.
COPY ["app", "./app"]

RUN pip install elasticsearch==7.12.1 fastapi==0.88.0 pyodbc==4.0.34 pyvi==0.1.1 sentence_transformers==2.2.2 tqdm==4.64.1 underthesea==1.3.5 urllib3==1.26.6 uvicorn==0.20.0 python-multipart

CMD ["python", "app/api.py"]