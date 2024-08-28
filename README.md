# PDF Embedding and Query System with LangChain and RAG

This project demonstrates how to build a system for creating and querying embeddings from PDF documents using LangChain, Chroma, and OpenAI. The system loads PDF files, splits the text into chunks, creates embeddings, and stores them in a vector database. It also provides functionality to query these embeddings for similarity and generate responses.

## Features

- **PDF Loading:** Load PDF documents from a specified directory.
- **Text Splitting:** Split the loaded text into manageable chunks.
- **Embedding Creation:** Create embeddings for the text chunks using OpenAI's models.
- **Vector Database Storage:** Store and persist the embeddings in a Chroma vector database.
- **Querying:** Search for similar content in the vector database and generate responses based on context.

## Prerequisites

- Python 3.7+
- LangChain
- Chroma
- OpenAI
- python-dotenv

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/pdf-embedding-query-system.git
    cd pdf-embedding-query-system
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the root directory of the project with your OpenAI API key (an example `.env.example` file is included in this repo):

    ```dotenv
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

### Create Database
Before running the command to create a database, you should create the two folders: `chroma/` and `data/`. The chroma folder will be used to store the embeddings in an sqlite database while the data folder will be where you will place all your pdfs. 

To create or update the vector database with embeddings from PDF files, use the `--createdb` flag:

```bash
python main.py --createdb
```
This command will:

- Load PDFs from the data directory.
- Split the text into chunks.
- Create embeddings and store them in the chroma directory.

### Query the Database
To query the vector database, provide a query text:

```bash
python main.py "Your query text here"
```

This command will:

Load the existing vector database from the chroma directory.
Search for similar content using the query.
Generate and print a response based on the context of the most relevant documents.

### Code Explanation

- `create_db()`: Loads documents, splits text, and saves embeddings to Chroma.
- `run_app(query_text)`: Queries the Chroma database and generates a response using OpenAI.

