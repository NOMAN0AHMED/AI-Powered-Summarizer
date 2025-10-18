# Yeh line os module import karta hai jo files aur system ke saath kaam karta hai
import os 
# dotenv se environment variables load karte hain, jaise API keys
from dotenv import load_dotenv
# Environment variables load karne ke liye
load_dotenv()
# LangChain ka ChatGroq model import kiya, jo AI ke saath baat karne ke liye hai
from langchain_groq import ChatGroq
# Streamlit import kiya, jo web app banane ke liye hai
import streamlit as st
# YouTube aur webpages se data lene ke liye loaders
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
# URL validate karne ke liye
import validators 
# Summarization chain ke liye
from langchain.chains.summarize import load_summarize_chain
# Prompt template banane ke liye
from langchain_core.prompts import ChatPromptTemplate
# Text ko chhote pieces mein divide karne ke liye
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Web app ka title jo screen par dikhega
st.title("AI-Powered Summarizer")
# Subheading jo batata hai ke yeh app kya karta hai
st.subheader("Enter a YouTube or Webpage URL, or paste text to get a summary")


# Sidebar banaya jahan extra controls hote hain
with st.sidebar:
    st.write("Sidebar ka content")  # Sidebar mein kuch text dikhaya
    # Groq API key input lene ke liye, password type mein chhupa hua
    groq_api_key = st.text_input("Apna Groq API key daalain", value="", type="password")

# User se input lene ke liye text area
user_input = st.text_area("URL ya Text daalain", label_visibility="collapsed")

# Jab summarize button dabaya jaye
if st.button("Summarize"):
    # Check karta hai ke API key aur input khali toh nahi
    if not groq_api_key.strip() or not user_input.strip():
        st.error("API key aur input (URL ya text) daalain")
    else:
        try:
            # ChatGroq model initialize kiya with LLaMA model
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

            # Check karta hai ke input URL hai ya direct text
            if validators.url(user_input):
            
                # Agar YouTube URL hai
                if "youtube.com" in user_input or "youtu.be" in user_input:
                    loader = YoutubeLoader.from_youtube_url(user_input, add_video_info=True)
                else:
                    # Agar normal webpage URL hai
                    loader = UnstructuredURLLoader(
                        urls=[user_input], ssl_verify=False,
                      # headers = {
                     #User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.188 Safari/537.36"
                     # }
                    )
                data = loader.load()  # Data load karta hai
            else:
                # Agar direct text hai toh usay document mein convert karta hai
                from langchain.schema import Document
                data = [Document(page_content=user_input)]

            # Text ko chhote pieces mein divide karta hai
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = splitter.split_documents(data)

            # Summarization ka prompt banaya
            refine_prompt = ChatPromptTemplate.from_messages([
                ("system", "Aap ek helpful assistant hain. 300 lafzon ka summary do."),
                ("human", "{text}")
            ])

            # Summarization chain banayi
            chain = load_summarize_chain(llm, chain_type="refine", refine_prompt=refine_prompt)
            result = chain.run(texts)  # Summary banaya
            st.success(result)  # Summary screen par dikhaya

        except Exception as e:
            # Agar koi error aaye toh dikhaya
            st.error(f"Error: {e}")