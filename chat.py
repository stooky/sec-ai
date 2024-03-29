import streamlit as st
import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings

import os
import requests

from dotenv import load_dotenv
load_dotenv()

LOCAL_HEADERS = {"User-Agent": "Joseph Conley me@jpc2.org"}

query_params = st.experimental_get_query_params()

# init session
if 'docs' not in st.session_state:
    st.session_state['docs'] = set()

if 'conversation' not in st.session_state:
    st.session_state.conversation = []

def autocomplete_query(query):
    url = "https://efts.sec.gov/LATEST/search-index"
    response = requests.request("GET", url, params={"keysTyped": query}, headers=LOCAL_HEADERS)
    hits = []
    for h in response.json().get("hits").get("hits"):
        hit = {"name": h.get("_source").get("entity"), "id": h.get("_id") }
        hits.append(hit)

    return hits

def get_filings(cik):
    padded_cik = str(cik).zfill(10)
    url = "https://data.sec.gov/submissions/CIK{}.json".format(padded_cik)
    response = requests.request("GET", url, headers={"User-Agent": "Joseph Conley me@jpc2.org"})
    filings = response.json().get("filings").get("recent")
    qa = [i for i, val in enumerate(filings.get("form")) if val in ["10-Q", "10-K"]]
    hits = [
            {"date": filings.get("filingDate")[i],
             "name": filings.get("primaryDocument")[i],
             "type": filings.get("form")[i],
             "url": 'https://www.sec.gov/Archives/edgar/data/' + cik + '/' + filings.get("accessionNumber")[i].replace("-", "") + '/' + filings.get("primaryDocument")[i]
             } for i in qa]

    return hits

# Function to handle user input and generate a response
def handle_question():
    query = st.session_state.user_input  # Capture user input
    if query:  # Check if the input is not empty
        docs = st.session_state['vectordb'].similarity_search(query)
        response = st.session_state['chain'].run(input_documents=docs, question=query)

        st.session_state.conversation.append(f"Me: {query}")  # Add user question to conversation
        st.session_state.conversation.append(f"Bot: {response}")  # Add bot response to conversation
        st.session_state.user_input = ""  # Clear input box after submission

st.title('SEC AI')

apiKey = query_params.get("apiKey", [""])[0]
st.text_input("OpenAI API Key", key="api_key", value=apiKey)
sec_input = st.text_input("Enter ticker/company name:", key="sec_input")
if len(sec_input) >= 3:
    suggestions = autocomplete_query(sec_input)
    selected_record = st.selectbox("Companies", suggestions, format_func=lambda x: x['name'])

    if selected_record:
        filings = get_filings(selected_record['id'])

        st.title("Results")
        for index, item in enumerate(filings):
            col1, col2, col3, col4 = st.columns([0.2, 0.2, 0.4, 0.2])
            with col1:
                st.write(item['date'])

            with col2:
                st.write(item['type'])

            with col3:
                st.write(item['name'])

            with col4:
                # Create a button for each row
                if st.button(f"Add Filing", key=item['url']):
                    st.session_state['docs'].add(item['url'])

    if 'docs' in st.session_state:
        st.write("Selected Docs:")
        for d in st.session_state['docs']:
            st.write(d)

    if st.button(f"Load Docs", key="start_chat"):
        #prep model
        chunks = []
        for d in st.session_state['docs']:
            loader = WebBaseLoader(d, header_template=LOCAL_HEADERS)
            docs = loader.load()

            # Split your website into big chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
            chunks.extend(text_splitter.split_documents(docs))

        embedding = OpenAIEmbeddings(openai_api_key=st.session_state["api_key"])
        st.session_state['vectordb'] = Chroma.from_documents(documents=chunks, embedding=embedding)

        llm = ChatOpenAI(temperature=0, openai_api_key=st.session_state["api_key"])
        st.session_state['chain'] = load_qa_chain(llm, chain_type="stuff")

    if 'chain' in st.session_state:
        st.text_area("Ask a question:", key="user_input", on_change=handle_question)
        for message in st.session_state.conversation:
            st.write(message)

else:
    st.write("No matching records found.")
