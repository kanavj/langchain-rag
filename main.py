# %%
import os
import dotenv

dotenv.load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
open_api_key = os.getenv("OPENAI_API_KEY")

import logging

logger = logging.getLogger("chatbot.log")
logger.setLevel(logging.DEBUG)

# %% [markdown]
# # Intialize vector database

# %%
import langchain_groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

llm = langchain_groq.ChatGroq(model="llama3-8b-8192")  # type: ignore
doc_path = "./documents/"

# %%
from langchain_huggingface import HuggingFaceEmbeddings
import torch

if torch.cuda.is_available():
    model_kwargs = {"device": "cuda"}
else:
    model_kwargs = {"device": "cpu"}

model_name = "BAAI/bge-large-en"
encode_kwargs = {"normalize_embeddings": True}
hf = Hugging(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# %%
vectorstore = Chroma(embedding_function=hf)
def updateVectorstore(vectstr:Chroma,files:list[str]):
    docs = []
    for doc in [_ for _ in files if str(_).endswith(".pdf")]:
        document = PyPDFLoader(doc)
        docs.append(document.load())
    for doc in [_ for _ in files if str(_).endswith(".csv")]:
        document = CSVLoader(doc)
        docs.append(document.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = []
    for doc in docs:
        split = text_splitter.split_documents(doc)
        splits.append(split)
    splits = sum(splits, [])
    if not len(splits):
        return
    vectstr.add_documents(documents=splits,
                        #   embedding_function=embedding_function
                          )
    return

# %%
updateVectorstore(vectorstore,[doc_path+_ for _ in os.listdir(doc_path)])

# %%
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

# %%
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# %% [markdown]
# # Creating response chains

# %% [markdown]
# ## Simple response chain

# %%
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
rag_chain_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)
rag_chain_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_docs)

# %% [markdown]
# ## Response chain with history

# %%
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

# %%
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# %%
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If the context doesn't provide the answer, say so and tell the answer according to your knowledge.\
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain_chat = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# %% [markdown]
# # Chat Bots

# %% [markdown]
# ## Simple response chat bot

# %%
import gradio as gr


# %%
# def ask(prompt: str):
#     resp = rag_chain_source.invoke(prompt)
#     return resp["answer"], resp["context"]


# demo = gr.Interface(
#     fn=ask,
#     inputs=["text"],
#     outputs=["text", "text"],
# )
# demo.launch(server_port=5000,inline=False)

# %% [markdown]
# ## Chat bot with history

# %%
def chat(message, history):
    chat_history = []
    if "files" in message and len(message["files"]) > 0:
        updateVectorstore(vectorstore,[_["path"] for _ in message["files"]])    
    for human, ai in history:
        if ai is not None and human is not None:
            chat_history.extend([HumanMessage(content=human), AIMessage(content=ai)])
    resp = rag_chain_chat.invoke({"input": message["text"], "chat_history": chat_history})
    return resp["answer"]

gr.ChatInterface(chat, multimodal=True).launch(server_port=5000, inline=False)
