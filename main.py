from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain import hub
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA


loader = WebBaseLoader("https://medium.com/@lets.see.1016/connect-google-bard-with-python-using-palm-api-5f1460fa6f68")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
all_splits = text_splitter.split_documents(data)

 # We can also try Ollama embeddings
vectorstore = Chroma.from_documents(documents=all_splits,
                                    embedding=OllamaEmbeddings())

QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")


llm = Ollama(model="llama2",
             verbose=True,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
# QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

while True:
    question = input("\nQuestion: ")
    result = qa_chain({"query": question})
    print(result)