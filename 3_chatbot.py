# import basics
import os
from dotenv import load_dotenv

# import streamlit
import streamlit as st

# import langchain / langgraph
from langgraph.prebuilt import create_react_agent
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

# load environment variables
load_dotenv()

##########################  INITIALIZE EMBEDDINGS MODEL  ######################################

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

##########################  INITIALIZE CHROMA VECTOR STORE  ###################################

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"),
)

##########################  INITIALIZE CHAT MODEL  ############################################

llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0
)

##########################  SYSTEM PROMPT  ####################################################

SYSTEM_PROMPT = """You are a helpful assistant.

You have access to a 'retrieve' tool that searches a knowledge base.

ONLY use the 'retrieve' tool when the user asks a specific question that requires factual information from the knowledge base.

Do NOT use the 'retrieve' tool for:
- Greetings (hi, hello, how are you)
- Simple conversational messages
- Questions you can answer from general knowledge

For every piece of information retrieved, provide the source URL.
If you don't know the answer after retrieving, say "I don't know" and don't provide a source.

Return your response as:
<Answer to the question>
Source: source_url (only if retrieve tool was used)
"""

##########################  RETRIEVER TOOL  ###################################################

@tool
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)

    serialized = ""
    for doc in retrieved_docs:
        print(f"DEBUG - Retrieved: {doc.metadata['source']} | {doc.page_content[:100]}")
        serialized += f"Source: {doc.metadata['source']}\nContent: {doc.page_content}\n\n"

    return serialized

tools = [retrieve]

##########################  INITIALIZE AGENT  #################################################

agent_executor = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

##########################  STREAMLIT APP  ####################################################

st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜")
st.title("🦜 Agentic RAG Chatbot")

# initialize chat history with system prompt only once
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(SYSTEM_PROMPT)]  # 👈 added only once

# display chat history (skip SystemMessage)
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# chat input
user_question = st.chat_input("Ask me anything...")

if user_question:
    # show user message
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append(HumanMessage(user_question))

    # invoke agent (system prompt already inside messages)
    result = agent_executor.invoke({"messages": st.session_state.messages})  # 👈 no duplicate system prompt
    ai_message = result["messages"][-1].content

    # show assistant response
    with st.chat_message("assistant"):
        st.markdown(ai_message)
    st.session_state.messages.append(AIMessage(ai_message))