# Update 29/03/2024: including source files in results

import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, CSVLoader, PyPDFLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_text_splitters import CharacterTextSplitter

import constants
import warnings


os.chdir("/Users/hngh7483/Dropbox/FoxAI/chatgpt")
os.environ["OPENAI_API_KEY"] = constants.APIKEY

loaders = {
    '.pdf': PyPDFLoader,
    '.csv': CSVLoader,
}

def create_directory_loader(file_type, directory_path):
    return DirectoryLoader(
        path=directory_path,
        glob=f"**/*{file_type}",
        loader_cls=loaders[file_type],
        loader_kwargs={"extract_images": False}
    )


# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False
query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  loader = create_directory_loader('.pdf', '/Users/hngh7483/Dropbox/FoxAI/chatgpt/data/pdf')
  #csv_loader = create_directory_loader('.csv', '/Users/hngh7483/Dropbox/chatgpt/data/csv')
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  else:
    #index = VectorstoreIndexCreator().from_loaders([loader])
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(loader.load_and_split())
    # load it into Chroma
    db = Chroma.from_documents(docs, OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=db)


memory = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True, output_key='answer')

chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=True,
        memory=memory)

    

chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  result = chain({"question": query, "chat_history": chat_history})
  print(result['answer'])
  query = None
