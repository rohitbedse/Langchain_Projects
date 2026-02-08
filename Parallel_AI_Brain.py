from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

# 1. LLM
llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# 2. Prompts
match_prompt = PromptTemplate.from_template("""
You are an ATS system.
Given Resume and Job Description, calculate skill match percentage (0-100).
Return ONLY JSON:
{{
  "match_percentage": number
}}

Resume:
{resume}

Job Description:
{jd}
""")

missing_prompt = PromptTemplate.from_template("""
You are a recruiter.
Find missing skills from resume compared to job description.
Return ONLY JSON:
{{
  "missing_skills": [ "skill1", "skill2" ]
}}

Resume:
{resume}

Job Description:
{jd}
""")

improve_prompt = PromptTemplate.from_template("""
You are a career coach.
Suggest improvements to the resume for this job.
Return ONLY JSON:
{{
  "improvement_suggestions": [ "point1", "point2" ]
}}

Resume:
{resume}

Job Description:
{jd}
""")

cover_prompt = PromptTemplate.from_template("""
You are an HR professional.
Write a short 3-line professional cover note for this job.
Return ONLY JSON:
{{
  "cover_note": "3 lines cover note"
}}

Resume:
{resume}

Job Description:
{jd}
""")

# 3. Output Parser
parser = JsonOutputParser()

# 4. Parallel Runnable Chain
parallel_chain = RunnableParallel({
    "match": match_prompt | llm | parser,
    "missing": missing_prompt | llm | parser,
    "improve": improve_prompt | llm | parser,
    "cover": cover_prompt | llm | parser,
})

# 5. Input
resume_text = """
I am a Python developer with experience in Pandas, NumPy, and basic Machine Learning.
Built small projects using Flask and REST APIs.
"""

jd_text = """
Looking for a Data Scientist with strong Python, Machine Learning, Deep Learning,
SQL, and experience with data visualization tools.
"""

# 6. Run
result = parallel_chain.invoke({
    "resume": resume_text,
    "jd": jd_text
})

# 7. Final Output
print("\n=== FINAL RESULT ===\n")
print(result)
