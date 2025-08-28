from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

model= ChatOpenAI(model= "gpt-4.1")
st.header("Research Assistant")

paper_input = st.selectbox("Select Research Paper Name", ["Attention is all you need", "BERT: Pre-training of Deep Biderectional Transformers", 
"GPT-3: Language Models are few shot learners" "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Begineer-Friendly", "Technical",
"Code-Oriented", "Mathematical"])

length_input= st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)",
"Medium(3-8 paragraphs)", "Long(detailed explanation)"])

# create template

templates= load_prompt("template.json")

# Fill the placeholder
prompt= templates.invoke({"paper_input":paper_input, "style_input": style_input, "length_input": length_input})

#user_input= st.text_input("Enter your prompt")

if st.button("Summarize"):
    result= model.invoke(prompt)
    #st.text("some random text")
    st.write(result.content)