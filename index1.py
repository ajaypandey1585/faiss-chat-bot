# Assume all required packages are installed
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# 1 & 2 Load the document and split into pages
loader = PyPDFLoader("KM_UseCase.pdf")
pages_content = loader.load_and_split()
#print(len(pages_content), pages_content)

# 3 Create embeddings and vector store
embeddings = OpenAIEmbeddings(OPENAPIKEY)
db = FAISS.from_documents(pages_content, embeddings)

# Save and load the vector store
db.save_local("faiss_index")
new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# 4 Perform query on the loaded vector store
query = "financial year commencing"
# docs = new_db.similarity_search(query)
# print(docs)

# 5 Set up the language model and query-answer chain
llm = ChatOpenAI(api_key="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")
qa_chain = RetrievalQA.from_chain_type(llm, retriever=new_db.as_retriever())
#res = qa_chain({"query": "What does the financial year commencing mean?"})
#print(res)

def ask(user_query):
    res = qa_chain({"query": user_query})
    return res["result"]