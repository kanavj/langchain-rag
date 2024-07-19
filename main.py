# %%
import os
import dotenv

dotenv.load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
open_api_key = os.getenv("OPENAI_API_KEY")

# %%
import langchain_groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

llm = langchain_groq.ChatGroq(model="llama3-8b-8192")
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# %%
doc_path = "./documents/"
docs = []
for doc in [_ for _ in os.listdir(doc_path) if str(_).endswith(".pdf")]:
    document = PyPDFLoader(doc_path + doc)
    docs.append(document.load())
for doc in [_ for _ in os.listdir(doc_path) if str(_).endswith(".csv")]:
    document = CSVLoader(doc_path + doc)
    docs.append(document.load())

# %%
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = []
for doc in docs:
    split = text_splitter.split_documents(doc)
    splits.append(split)
splits = sum(splits, [])

# %%
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_function,
    # embedding_function = OpenAIEmbeddings(),
)

# %%
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


# %%
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# %%
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# # %%
# ans = rag_chain.invoke(
#     "Could you tell me the average age of the students in the Student performance data file?"
# )
# print(str(ans))

# %%
import gradio as gr


def ask(prompt: str):
    return rag_chain.invoke(prompt)


demo = gr.Interface(
    fn=ask,
    inputs=["text"],
    outputs=["text"],
)

# %%
demo.launch()
