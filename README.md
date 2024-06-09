# Avalok AI
Avalok is a Sanskrit word meaning Search.

Avalok AI is a platform to build custom search on your documents and build AI assistants to answer your questions from those documents. Avalok AI is highly scalable and easily be integrated in your software to power your search.

It provides a comprehensive solution for building search and RAG applications so that you do not have to spend time in building these solutions from scratch

# Features
- Directly search inside documents on S3.
- Use with OpenAI, Gemini APIs or run locally with Hugging Face models and Sentence Transformers.
- Self hosted Vector database or Managed Database
- Connect any data source for indexing.
- UI to search for documents.

# Tech Stack
- Pytorch for running the models locally.
- Langchain for splitting the documents.
- ChromaDB as vector store.
- Celery for task quequing. Rabbit MQ as the message service.
- Streamlit for UI.

# Getting Started

Avalok AI currently sopports build from source only. We recommend using conda or virtual env to install Avalok AI.

```bash
git clone https://github.com/AvalokAI/AvalokAI.git
cd AvalokAI
conda create -n deepsearch python=3.10
conda activate deepsearch
pip install -r requirements.txt
sudo apt-get install rabbitmq-server
pip install -e .
```

This installs Avalok AI. It contains two modules `avalokai` and `avalokui` for building search system and for viewing search.

## Additional steps for S3 indexing

If you want to index documents stored in S3, you need to setup AWS keys. 

- Install and configure awscli and run aws configure.
- Set credentials in the AWS credentials profile file on the local system, located at: ~/.aws/credentials on Unix or macOS.
- Set the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables.

Setup instructions can also be found on https://github.com/awslabs/s3-connector-for-pytorch

## Additional Steps to setup OpenAI and Gemini

If you want to use OpenAI or Gemini Models, add your keys in the environment variable in `OPENAI_API_KEY` and `GEMINI_API_KEY` respectively.

## Indexing Documents
Before indexing we need to start our vector database and celery queue.
### Start ChromaDB
In a new terminal, start chroma db for storing the vectors
```bash
mkdir chromadb_database
cd chromadb_database
chroma run --path ./
```
### Start Celery Queue
In a new terminal, start the celery queue.
```bash
cd AvalokAI/avalokai
celery -A sink worker -l INFO --concurrency=4
```
### Index S3 bucket

Using the below code you can start indexing the documents. You can also index custom documents. Instructions can be found at [below](#indexing-custom-documents)

```python
from avalokai import Indexer
BUCKET = "s3://your-bucket"
REGION = "us-east-1"
DBNAME = "your-database-name"
indexer = Indexer(DBNAME, './config.yaml')
indexer.index_s3_data(BUCKET, REGION)
```
### Searching for documents

You can search for documents using the below script.

```python
from avalokai import Searcher
DBNAME = "your-database-name"
searcher = Searcher(dbname, './config.yaml')
query = input("Enter your query: ")
matches = searcher.search(query, top_k=10)
for _, match in enumerate(matches):
    doc_id = match["id"]
    print(f"Document id {doc_id},  document match score {match['score']}"),
    print(f"{match['content']}")
```
### Running UI for searching
You can also run the streamlit app for searching the documents.
Create a file `app.py`
```python
from avalokui import display
DBNAME = "your-database-name"
def main():
    display(DBNAME, "./config.yaml")
if __name__ == "__main__":
    main()
```
Run the file using `streamlit run app.py`

## Config File
Indexing strategies are controlled using `config.yaml`. We give a default `config.yaml`. You can edit the config to include custom models.

Config is divided into two parts, main config and model config.

We provide `model_name` in the main_config to use that model for indexing. Batch size controls the number of samples loaded in each iteration for indexing.

Each model config starts with model name and includes several parameters. `model_type` in the config specified the type of model such as openai or hugging face so that code can handle that model.

## Indexing custom documents.
We make it easy to index any custom document which does not fit the pre built indexing methods.

The basic idea is to convert each document into `RawData` format which avalok can understand. Each `RawData` object should contain a unique id, content, and optional metadata. Basic example is shown here
```python
from avalokai import Indexer
from avalokai.source import RawData

DBNAME = "your-database-name"
indexer = Indexer(DBNAME, './config.yaml')

document1 = RawData('id1','content1')
document2 = RawData('id2','content2')

indexer.index_raw_data([document1, document2])
```

# Features in development
- [ ] Batch processing in OpenAI
- [ ] Add support for multiple vector databases like Pinecone.
- [ ] Add support for basic search algorithms such as BM25
- [ ] Add support for indexing only changed files in S3.
- [ ] Add more data sources.

# Feature request 
- Please start a discussion or github issue to request for a feature.

# Feedback 
We are always open for feedback. Start a github discussion or github issue. You can also email us at avalokai@gmail.com

Consider signing up for updates at [https://avalokai.com](https://avalokai.wixsite.com/avalokai)
