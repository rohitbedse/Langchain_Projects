# filename: linkedin_caption_gemini_streamlit.py
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# Initialize HF endpoint (Gemini 2.5 Flash or Zephyr)
llm_endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.3,
)
model = ChatHuggingFace(llm=llm_endpoint)

# Step 1: Draft prompt (human-friendly + style variable)
template1 = PromptTemplate(
    template="""
You are an experienced LinkedIn content writer. 
Write a professional, friendly, and engaging LinkedIn caption based on the topic below. 
Make it feel personal and human-written.

Topic: {topic}
Style: {style}

Guidelines:
- Keep it concise (3-5 sentences)
- Invite readers to connect or collaborate
- Add 5-7 relevant, natural hashtags
- Do not repeat hashtags
- Make it sound like a real human wrote it
""",
    input_variables=["topic", "style"],
)

# Step 2: Refinement prompt
template2 = PromptTemplate(
    template="Refine the following text to match style: {style}\n\nText:\n{text}",
    input_variables=["text", "style"],
)

parser = StrOutputParser()

# RunnableLambda to convert draft string to dict expected by template2
wrap_for_refine = RunnableLambda(lambda draft, style: {"text": draft, "style": style})

# Streamlit UI
st.title("LinkedIn Caption Generator 🚀")

topic_input = st.text_area("Enter your post topic / idea:")
style_option = st.selectbox("Choose caption style:", ["Official", "Exciting", "No-fluff", "Storytelling"])

if st.button("Generate Caption"):
    if not topic_input.strip():
        st.warning("Please enter some content first!")
    else:
        with st.spinner("Generating caption..."):
            # Build the chain dynamically with style bound
            chain = (
                template1
                | model
                | parser
                | wrap_for_refine.bind(style=style_option)
                | template2
                | model
                | parser
            )
            # Pass both topic and style for template1
            result = chain.invoke({"topic": topic_input, "style": style_option})
        st.subheader("Generated Caption:")
        st.success(result)
