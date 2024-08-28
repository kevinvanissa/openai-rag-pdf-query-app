import argparse
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

DATA_PATH="data"
CHROMA_PATH = "chroma"
PROMT_TEMPLATE = """
Answer the following based only on the following context:
{context}

---

Answer the question based on the above context: {question}
"""


def create_db():
    docs = _loadDocuments()
    chunks = _splitText(docs)
    _saveToChroma(chunks)


def _loadDocuments():
    pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
    data = pdf_loader.load()
    return data


def _splitText(docs):
    text_splitter = RecursiveCharacterTextSplitter(
           chunk_size=500,
           chunk_overlap = 50,
           length_function=len,
           is_separator_regex=False,
            )
    chunks = text_splitter.split_documents(docs)
    return chunks


def _saveToChroma(chunks):
    # remove if exist
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Use OpenAIEmbeddings to create embeddings from document
    db = Chroma.from_documents(
            chunks,
            OpenAIEmbeddings(),
            persist_directory=CHROMA_PATH
            )
    #just to make sure data is saved
    db.persist()
    print(f"{len(chunks)} chunks saved.")


def run_app(query_text):
    # initialize the DB
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the database for similarity 
    results = db.similarity_search_with_relevance_scores(query, k=3)
    if not results or results[0][1] < 0.7:
        print("No Matching Result")
        return

    context_text = "\n\n****\n\n".join([ doc.page_content for doc, _ in results ])
    prompt_template = ChatPromptTemplate.from_template(PROMT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(f"Printing Prompt: \n--------------------\n {prompt}")

    model = ChatOpenAI()
    response = model.invoke(prompt)

    # Show sources and response
    sources = [doc.metadata.get("source", "") for doc, _ in results]
    print(f"Response: {response} \n\nSources: {sources}") 


if __name__ == "__main__":
    #run_app()
    parser = argparse.ArgumentParser(description="App to query bible lessons")
    parser.add_argument("query", type=str, nargs="?", help="The text for querying")
    parser.add_argument("--createdb", action="store_true", help="Create vector database embeddings")
    args = parser.parse_args()

    if args.createdb:
        print("Creating Database")
        create_db()
    elif args.query:
        query = args.query
        print("Querying App")
        run_app(query)
    else:
        print("No arguments provided")
