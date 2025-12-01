from dotenv import load_dotenv
import streamlit as st

# LangChain LLMs
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

# load the env vars
load_dotenv()

# -------------------------
# PROVIDER → MODEL OPTIONS
# -------------------------
provider_models = {
    "OpenAI": ["gpt-4.1", "gpt-4o-mini"],
    "Gemini": ["gemini-2.0-flash", "gemini-2.0-pro"],
    "Groq": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
    "Ollama": ["llama3.1", "gemma2:2b"]
}

# streamlit page setup   # for emoji use website emojidb.org
st.set_page_config(
    page_title="Multi-Model Chatbot", 
    page_icon="🤖", 
    layout="centered",
 )
st.title("💬 Multi-Model Generative AI Chatbot")

# Initiate chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
    
    
# -------------------------
# SELECT PROVIDER
# -------------------------
provider = st.selectbox("Choose LLM provider:", list(provider_models.keys()))

# SELECT MODEL
model_name = st.selectbox("Choose model:", provider_models[provider])


# -------------------------
# FUNCTION: RETURN LLM BASED ON SELECTION
# -------------------------
def get_llm(provider, model_name):
    if provider == "OpenAI":
        return ChatOpenAI(model=model_name, temperature=0)

    if provider == "Gemini":
        return ChatGoogleGenerativeAI(model=model_name, temperature=0)

    if provider == "Groq":
        return ChatGroq(model=model_name, temperature=0)

    if provider == "Ollama":
        return ChatOllama(model=model_name, temperature=0)

# Get LLM instance
llm = get_llm(provider, model_name)


# show chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# input box 
user_prompt = st.chat_input("Ask Chatbot...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        
    response = llm.invoke(
        input=[{"role": "system", "content": "You are helpful assistant"}, *st.session_state.chat_history]
    )
    asisstant_response = response.content
    st.session_state.chat_history.append({"role": "assistant", "content": asisstant_response})
    
    with st.chat_message("assistant"):
        st.markdown(asisstant_response)