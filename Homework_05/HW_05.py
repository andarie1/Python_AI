from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def summarize_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)

summary = summarize_url("https://habr.com/ru/articles/836464/")
print(summary)