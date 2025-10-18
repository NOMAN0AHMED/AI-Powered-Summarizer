import os 

from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq

import streamlit as st

from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

import validators 

from langchain.chains.summarize import load_summarize_chain

from langchain_core.prompts import ChatPromptTemplate

from langchain.text_splitter import RecursiveCharacterTextSplitter


st.title("AI-Powered Summarizer")

st.subheader("Enter a YouTube or Webpage URL, or paste text to get a summary")



with st.sidebar:
    st.write("Sidebar ka content")  
    groq_api_key = st.text_input("Apna Groq API key daalain", value="", type="password")


user_input = st.text_area("URL ya Text daalain", label_visibility="collapsed")

if st.button("Summarize"):

    if not groq_api_key.strip() or not user_input.strip():
        st.error("API key aur input (URL ya text) daalain")
    else:
        try:
       
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)


            if validators.url(user_input):
            
  
                if "youtube.com" in user_input or "youtu.be" in user_input:
                    loader = YoutubeLoader.from_youtube_url(user_input, add_video_info=True)
                else:
          
                    loader = UnstructuredURLLoader(
                        urls=[user_input], ssl_verify=False,
                      # headers = {
                     #User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.188 Safari/537.36"
                     # }
                    )
                data = loader.load() 
            else:
               
                from langchain.schema import Document
                data = [Document(page_content=user_input)]

       
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = splitter.split_documents(data)

        
            refine_prompt = ChatPromptTemplate.from_messages([
                ("system", "Aap ek helpful assistant hain. 300 lafzon ka summary do."),
                ("human", "{text}")
            ])

          
            chain = load_summarize_chain(llm, chain_type="refine", refine_prompt=refine_prompt)
            result = chain.run(texts)  
            st.success(result)  

        except Exception as e:
          
            st.error(f"Error: {e}")
