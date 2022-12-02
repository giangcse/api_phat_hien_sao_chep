FROM laudio/pyodbc:1.0.38

WORKDIR /source

# Add your source files.
COPY [".", "./app"]

RUN pip install elasticsearch==7.12.* fastapi==0.88.* pyodbc==4.0.* pyvi==0.1.* sentence_transformers==2.2.* tqdm==4.64.* underthesea==1.3.5 urllib3==1.26.* uvicorn==0.20.* python-multipart

CMD ["python", "app/api.py"]