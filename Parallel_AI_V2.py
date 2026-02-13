import streamlit as st
import os
import json
import re
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ============== CONFIGURATION ==============

class Theme:
    """Cyberpunk/Neon dark theme optimized for focus and reduced eye strain"""
    BG_PRIMARY = "#0f172a"      # Deep slate
    BG_SECONDARY = "#1e293b"    # Card background
    BG_TERTIARY = "#334155"     # Elevated elements
    ACCENT_PRIMARY = "#06b6d4"  # Cyan neon
    ACCENT_SECONDARY = "#8b5cf6" # Violet neon
    ACCENT_SUCCESS = "#10b981"  # Emerald
    ACCENT_WARNING = "#f59e0b"  # Amber
    ACCENT_DANGER = "#ef4444"   # Red
    TEXT_PRIMARY = "#f8fafc"    # White
    TEXT_SECONDARY = "#94a3b8"  # Muted gray
    TEXT_ACCENT = "#38bdf8"     # Light blue
    GRADIENT_START = "#06b6d4"
    GRADIENT_END = "#8b5cf6"

# ============== DATA MODELS ==============

class SkillCategory(str, Enum):
    TECHNICAL = "technical"
    SOFT = "soft"
    DOMAIN = "domain"
    TOOLS = "tools"

class SkillGap(BaseModel):
    skill: str
    importance: int = Field(1, ge=1, le=5)
    category: SkillCategory
    learning_resources: List[str] = Field(default_factory=list)
    estimated_hours: int = 0

class ExperienceMatch(BaseModel):
    role_similarity: float = Field(0, ge=0, le=100)
    industry_alignment: float = Field(0, ge=0, le=100)
    seniority_match: float = Field(0, ge=0, le=100)
    achievement_relevance: float = Field(0, ge=0, le=100)

class CareerInsight(BaseModel):
    match_score: float = Field(0, ge=0, le=100)
    experience_analysis: ExperienceMatch
    skill_gaps: List[SkillGap]
    competitive_advantages: List[str]
    market_position: str  # "Entry", "Mid", "Senior", "Expert"
    salary_range_estimate: str
    career_trajectory: List[str]  # Next 3 potential roles

class ResumeAnalysis(BaseModel):
    ats_compatibility_score: float = Field(0, ge=0, le=100)
    keyword_optimization: float = Field(0, ge=0, le=100)
    formatting_score: float = Field(0, ge=0, le=100)
    content_quality: float = Field(0, ge=0, le=100)
    improvement_suggestions: List[Dict[str, str]]  # category: suggestion
    optimized_bullet_points: List[str]

class CoverLetter(BaseModel):
    hook: str
    body_paragraphs: List[str]
    closing: str
    keywords_included: List[str]
    tone_analysis: str

class InterviewPrep(BaseModel):
    likely_questions: List[Dict[str, str]]  # question: suggested_answer
    technical_challenges: List[str]
    behavioral_scenarios: List[str]
    questions_to_ask: List[str]

class CareerIntelligenceOutput(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True,
        "populate_by_name": True
    }
    
    career_insight: CareerInsight
    resume_analysis: ResumeAnalysis
    cover_letter: CoverLetter
    interview_prep: InterviewPrep
    skill_development_roadmap: List[Dict[str, Any]]
    networking_strategy: List[str]

# ============== CORE INTELLIGENCE ENGINE ==============

class CareerIntelligenceEngine:
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            convert_system_message_to_human=True
        )
        self.structured_llm = self.llm.with_structured_output(
            CareerIntelligenceOutput,
            method="json_mode"
        )
    
    def analyze(self, resume: str, job_description: str, student_context: Dict) -> CareerIntelligenceOutput:
        system_prompt = """You are an elite Career Intelligence AI specializing in student career development.
        Analyze the resume against the job description with extreme precision, focusing on:
        1. Hidden skill transfers (academic projects → professional skills)
        2. Growth trajectory potential (not just current state)
        3. Market positioning strategy for entry-level candidates
        4. Specific, actionable intelligence (not generic advice)"""
        
        user_prompt = f"""
        RESUME:
        {resume}
        
        JOB DESCRIPTION:
        {job_description}
        
        STUDENT CONTEXT:
        - Academic Level: {student_context.get('level', 'Undergraduate')}
        - Field of Study: {student_context.get('field', 'Not specified')}
        - Graduation Date: {student_context.get('graduation', 'Unknown')}
        - Career Goals: {student_context.get('goals', 'Not specified')}
        - Previous Internships: {student_context.get('internships', 'None')}
        
        Provide comprehensive career intelligence including:
        1. Realistic match score with detailed breakdown
        2. Hidden skill gaps that aren't obvious
        3. Competitive advantages unique to this candidate
        4. ATS optimization specific to this industry
        5. Custom cover letter strategy
        6. Interview preparation with likely technical and behavioral questions
        7. 30-60-90 day skill development roadmap
        8. Strategic networking approach
        
        Be specific, data-driven, and actionable. Avoid generic advice."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        return self.structured_llm.invoke(messages)

# ============== VISUALIZATION COMPONENTS ==============

def create_gauge_chart(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24, 'color': Theme.TEXT_PRIMARY}},
        number={'font': {'size': 40, 'color': color}, 'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': Theme.TEXT_SECONDARY},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': Theme.BG_TERTIARY,
            'borderwidth': 2,
            'bordercolor': Theme.BG_SECONDARY,
            'steps': [
                {'range': [0, 50], 'color': f"rgba(239, 68, 68, 0.2)"},
                {'range': [50, 75], 'color': f"rgba(245, 158, 11, 0.2)"},
                {'range': [75, 100], 'color': f"rgba(16, 185, 129, 0.2)"}
            ],
            'threshold': {
                'line': {'color': Theme.TEXT_PRIMARY, 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': Theme.TEXT_PRIMARY},
        margin=dict(l=20, r=20, t=50, b=20),
        height=300
    )
    return fig

def create_radar_chart(categories: List[str], values: List[float]) -> go.Figure:
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(6, 182, 212, 0.3)',
        line=dict(color=Theme.ACCENT_PRIMARY, width=3),
        name='Candidate Profile'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color=Theme.TEXT_SECONDARY),
                gridcolor=Theme.BG_TERTIARY
            ),
            angularaxis=dict(
                tickfont=dict(color=Theme.TEXT_PRIMARY, size=12),
                gridcolor=Theme.BG_TERTIARY
            ),
            bgcolor=Theme.BG_SECONDARY
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=80, r=80, t=40, b=40),
        height=400
    )
    return fig

def create_skill_gap_chart(skill_gaps: List[SkillGap]) -> go.Figure:
    skills = [gap.skill for gap in skill_gaps[:8]]  # Top 8
    importance = [gap.importance for gap in skill_gaps[:8]]
    hours = [gap.estimated_hours for gap in skill_gaps[:8]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=skills,
        x=importance,
        name='Importance',
        orientation='h',
        marker=dict(color=Theme.ACCENT_DANGER, opacity=0.8),
        text=importance,
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        y=skills,
        x=[h/10 for h in hours],  # Scale hours for visibility
        name='Learning Hours (÷10)',
        orientation='h',
        marker=dict(color=Theme.ACCENT_SUCCESS, opacity=0.6),
        text=hours,
        textposition='auto',
    ))
    
    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=Theme.BG_SECONDARY,
        font=dict(color=Theme.TEXT_PRIMARY),
        xaxis=dict(title='Priority Level', tickfont=dict(color=Theme.TEXT_SECONDARY)),
        yaxis=dict(tickfont=dict(color=Theme.TEXT_PRIMARY)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=150, r=20, t=60, b=40),
        height=400
    )
    return fig

# ============== STREAMLIT APP ==============

st.set_page_config(
    page_title="CareerSync AI | Student Career Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cyberpunk theme with text wrapping fixes
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    * {{
        font-family: 'Space Grotesk', sans-serif;
    }}
    
    .stApp {{
        background: linear-gradient(135deg, {Theme.BG_PRIMARY} 0%, #020617 100%);
    }}
    
    .main-header {{
        background: linear-gradient(135deg, {Theme.ACCENT_PRIMARY} 0%, {Theme.ACCENT_SECONDARY} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
    }}
    
    .sub-header {{
        color: {Theme.TEXT_SECONDARY};
        text-align: center;
        font-size: clamp(1rem, 2vw, 1.2rem);
        margin-bottom: 3rem;
        font-weight: 300;
    }}
    
    .metric-card {{
        background: linear-gradient(135deg, {Theme.BG_SECONDARY} 0%, {Theme.BG_TERTIARY} 100%);
        border: 1px solid rgba(6, 182, 212, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.1);
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(6, 182, 212, 0.2);
        border-color: {Theme.ACCENT_PRIMARY};
    }}
    
    .neon-text {{
        color: {Theme.ACCENT_PRIMARY};
        text-shadow: 0 0 10px rgba(6, 182, 212, 0.5);
    }}
    
    .insight-box {{
        background: {Theme.BG_SECONDARY};
        border-left: 4px solid {Theme.ACCENT_PRIMARY};
        padding: 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }}
    
    .skill-tag {{
        display: inline-block;
        background: rgba(139, 92, 246, 0.2);
        color: {Theme.ACCENT_SECONDARY};
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        margin: 0.25rem;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }}
    
    .advantage-tag {{
        display: inline-block;
        background: rgba(16, 185, 129, 0.2);
        color: {Theme.ACCENT_SUCCESS};
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        margin: 0.25rem;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }}
    
    .roadmap-item {{
        background: {Theme.BG_SECONDARY};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid {Theme.BG_TERTIARY};
        position: relative;
        overflow: hidden;
        word-wrap: break-word;
    }}
    
    .roadmap-item::before {{
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 4px;
        background: linear-gradient(180deg, {Theme.ACCENT_PRIMARY}, {Theme.ACCENT_SECONDARY});
    }}
    
    .stButton>button {{
        background: linear-gradient(135deg, {Theme.ACCENT_PRIMARY} 0%, {Theme.ACCENT_SECONDARY} 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3);
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(6, 182, 212, 0.5);
    }}
    
    .stTextArea>div>div>textarea, .stTextInput>div>div>input {{
        background: {Theme.BG_SECONDARY};
        color: {Theme.TEXT_PRIMARY};
        border: 1px solid {Theme.BG_TERTIARY};
        border-radius: 12px;
        font-family: 'JetBrains Mono', monospace;
    }}
    
    .stSelectbox>div>div, .stRadio>div {{
        background: {Theme.BG_SECONDARY};
        border-radius: 12px;
    }}
    
    .section-title {{
        color: {Theme.TEXT_PRIMARY};
        font-size: clamp(1.2rem, 3vw, 1.8rem);
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .highlight-box {{
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }}
    
    .exec-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        text-align: center;
    }}
    
    @media (max-width: 992px) {{
        .exec-grid {{
            grid-template-columns: repeat(2, 1fr);
        }}
    }}
    
    @media (max-width: 576px) {{
        .exec-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    
    .metric-value {{
        font-size: clamp(1.5rem, 4vw, 2.5rem);
        font-weight: 700;
        word-break: break-word;
        line-height: 1.2;
    }}
    
    .metric-label {{
        color: {Theme.TEXT_SECONDARY};
        font-size: clamp(0.8rem, 1.5vw, 0.9rem);
        margin-top: 0.25rem;
    }}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {{
        width: 10px;
        background: {Theme.BG_PRIMARY};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {Theme.BG_TERTIARY};
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {Theme.ACCENT_PRIMARY};
    }}
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

# Header
st.markdown('<h1 class="main-header">🎯 CareerSync AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Next-Gen Career Intelligence for Ambitious Students</p>', unsafe_allow_html=True)

# Sidebar - Student Context
with st.sidebar:
    st.markdown(f"""
    <div style="background: {Theme.BG_SECONDARY}; padding: 1.5rem; border-radius: 16px; margin-bottom: 1rem;">
        <h3 style="color: {Theme.ACCENT_PRIMARY}; margin-bottom: 1rem;">👤 Student Profile</h3>
    """, unsafe_allow_html=True)
    
    student_level = st.selectbox(
        "Academic Level",
        ["Freshman", "Sophomore", "Junior", "Senior", "Graduate Student", "Recent Graduate"],
        index=3
    )
    
    field_of_study = st.text_input(
        "Field of Study",
        placeholder="e.g., Computer Science, Business Administration",
        value="Computer Science"
    )
    
    graduation_date = st.date_input(
        "Expected Graduation",
        value=datetime(2025, 6, 1)
    )
    
    career_goals = st.text_area(
        "Career Goals",
        placeholder="What roles are you targeting? What companies interest you?",
        value="Software Engineering at top tech companies"
    )
    
    internships = st.text_area(
        "Previous Experience",
        placeholder="List any internships, projects, or relevant experience",
        value="Summer internship at startup, built ML project for class"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis trigger
    st.markdown("---")
    analyze_btn = st.button("🔮 Generate Career Intelligence", use_container_width=True)
    
    if st.session_state.analyzed:
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.analyzed = False
            st.session_state.analysis_result = None
            st.rerun()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<h3 class="section-title">📄 Paste Your Resume</h3>', unsafe_allow_html=True)
    resume_text = st.text_area(
        "",
        height=300,
        placeholder="Paste your resume content here... Include: Education, Skills, Projects, Experience, Achievements",
        value="""Education:
- BS Computer Science, GPA 3.7
- Relevant Coursework: Data Structures, Algorithms, Machine Learning

Skills:
- Python, JavaScript, SQL
- React, Node.js, Flask
- Basic TensorFlow, Pandas, NumPy

Projects:
- Built web app for local business using React and Firebase
- Machine learning model for sentiment analysis (85% accuracy)
- Automated data pipeline for research lab

Experience:
- Teaching Assistant for Intro to Programming
- Freelance web developer (3 projects)"""
    )

with col2:
    st.markdown('<h3 class="section-title">💼 Target Job Description</h3>', unsafe_allow_html=True)
    jd_text = st.text_area(
        "",
        height=300,
        placeholder="Paste the job description here...",
        value="""Software Engineer - Entry Level

Requirements:
- BS in Computer Science or related field
- Strong Python and JavaScript skills
- Experience with React and modern web frameworks
- Knowledge of SQL and database design
- Understanding of REST APIs and microservices
- Familiarity with cloud platforms (AWS/GCP)
- Machine learning experience is a plus
- Strong problem-solving skills
- Excellent communication and teamwork

Responsibilities:
- Develop and maintain web applications
- Collaborate with cross-functional teams
- Write clean, scalable code
- Participate in code reviews
- Learn and adapt to new technologies"""
    )

# Analysis execution
if analyze_btn and resume_text and jd_text:
    with st.spinner("🧠 Analyzing career trajectory..."):
        try:
            engine = CareerIntelligenceEngine()
            
            student_context = {
                "level": student_level,
                "field": field_of_study,
                "graduation": graduation_date.strftime("%B %Y"),
                "goals": career_goals,
                "internships": internships
            }
            
            result = engine.analyze(resume_text, jd_text, student_context)
            st.session_state.analysis_result = result
            st.session_state.analyzed = True
            st.rerun()
            
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")

# Display results
if st.session_state.analyzed and st.session_state.analysis_result:
    result = st.session_state.analysis_result
    
    # Executive Summary - FIXED LAYOUT
    st.markdown(f"""
    <div class="highlight-box">
        <h2 style="color: {Theme.ACCENT_PRIMARY}; margin-bottom: 1.5rem; font-size: clamp(1.5rem, 4vw, 2rem);">
            🎯 Executive Summary
        </h2>
        <div class="exec-grid">
            <div class="metric-card">
                <div class="metric-value" style="color: {Theme.ACCENT_PRIMARY};">
                    {result.career_insight.match_score:.0f}%
                </div>
                <div class="metric-label">Match Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: {Theme.ACCENT_SECONDARY};">
                    {result.career_insight.market_position}
                </div>
                <div class="metric-label">Market Position</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: {Theme.ACCENT_SUCCESS};">
                    {len(result.career_insight.skill_gaps)}
                </div>
                <div class="metric-label">Skill Gaps</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: {Theme.ACCENT_WARNING};">
                    {result.career_insight.salary_range_estimate}
                </div>
                <div class="metric-label">Est. Salary</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Charts Row
    chart_col1, chart_col2 = st.columns([1, 1])
    
    with chart_col1:
        st.plotly_chart(
            create_gauge_chart(
                result.career_insight.match_score,
                "Overall Match",
                Theme.ACCENT_PRIMARY if result.career_insight.match_score > 70 else Theme.ACCENT_WARNING
            ),
            use_container_width=True
        )
    
    with chart_col2:
        categories = ['Role Fit', 'Industry', 'Seniority', 'Achievements']
        values = [
            result.career_insight.experience_analysis.role_similarity,
            result.career_insight.experience_analysis.industry_alignment,
            result.career_insight.experience_analysis.seniority_match,
            result.career_insight.experience_analysis.achievement_relevance
        ]
        st.plotly_chart(create_radar_chart(categories, values), use_container_width=True)
    
    # Detailed Analysis Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Career Strategy", 
        "⚡ Skill Intelligence", 
        "📝 Resume Optimization",
        "🎤 Interview Prep"
    ])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<h3 class="section-title">Competitive Advantages</h3>', unsafe_allow_html=True)
            for advantage in result.career_insight.competitive_advantages:
                st.markdown(f'<span class="advantage-tag">✨ {advantage}</span>', unsafe_allow_html=True)
            
            st.markdown('<h3 class="section-title" style="margin-top: 2rem;">Career Trajectory</h3>', unsafe_allow_html=True)
            for i, role in enumerate(result.career_insight.career_trajectory, 1):
                st.markdown(f"""
                <div class="roadmap-item">
                    <strong style="color: {Theme.ACCENT_PRIMARY};">Step {i}:</strong> {role}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<h3 class="section-title">Networking Strategy</h3>', unsafe_allow_html=True)
            # FIXED: Use result.networking_strategy instead of result.career_insight.networking_strategy
            for strategy in result.networking_strategy:
                st.markdown(f"""
                <div class="insight-box">
                    {strategy}
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.plotly_chart(create_skill_gap_chart(result.career_insight.skill_gaps), use_container_width=True)
        
        st.markdown('<h3 class="section-title">Priority Skill Development</h3>', unsafe_allow_html=True)
        for gap in result.career_insight.skill_gaps[:5]:
            with st.expander(f"📚 {gap.skill} (Importance: {gap.importance}/5)"):
                st.markdown(f"**Category:** {gap.category.value}")
                st.markdown(f"**Estimated Learning Time:** {gap.estimated_hours} hours")
                st.markdown("**Resources:**")
                for resource in gap.learning_resources:
                    st.markdown(f"- {resource}")
    
    with tab3:
        cols = st.columns(2)
        with cols[0]:
            st.plotly_chart(
                create_gauge_chart(result.resume_analysis.ats_compatibility_score, "ATS Score", Theme.ACCENT_SUCCESS),
                use_container_width=True
            )
        with cols[1]:
            st.plotly_chart(
                create_gauge_chart(result.resume_analysis.keyword_optimization, "Keywords", Theme.ACCENT_SECONDARY),
                use_container_width=True
            )
        
        st.markdown('<h3 class="section-title">AI-Optimized Bullet Points</h3>', unsafe_allow_html=True)
        for bullet in result.resume_analysis.optimized_bullet_points[:5]:
            st.markdown(f"""
            <div class="insight-box" style="border-left-color: {Theme.ACCENT_SUCCESS};">
                <code style="color: {Theme.TEXT_PRIMARY};">{bullet}</code>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<h3 class="section-title">Critical Improvements</h3>', unsafe_allow_html=True)
        for suggestion in result.resume_analysis.improvement_suggestions[:5]:
            for category, text in suggestion.items():
                st.warning(f"**{category}:** {text}")
    
    with tab4:
        st.markdown('<h3 class="section-title">Likely Interview Questions</h3>', unsafe_allow_html=True)
        for qa in result.interview_prep.likely_questions[:5]:
            for question, answer in qa.items():
                with st.expander(f"❓ {question[:80]}..."):
                    st.markdown(f"**Suggested Answer:** {answer}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<h4 style="color: {Theme.ACCENT_WARNING};">Technical Challenges</h4>', unsafe_allow_html=True)
            for challenge in result.interview_prep.technical_challenges[:3]:
                st.markdown(f"- {challenge}")
        
        with col2:
            st.markdown(f'<h4 style="color: {Theme.ACCENT_SECONDARY};">Questions to Ask Them</h4>', unsafe_allow_html=True)
            for question in result.interview_prep.questions_to_ask[:3]:
                st.markdown(f"- {question}")

else:
    # Empty state with features
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; margin: 2rem 0; border: 2px dashed #334155; border-radius: 20px;">
        <h2 style="color: #94a3b8; margin-bottom: 1rem;">🚀 Ready to Unlock Your Career Potential?</h2>
        <p style="color: #64748b; max-width: 600px; margin: 0 auto 2rem auto;">
            Paste your resume and target job description, then click "Generate Career Intelligence" 
            to receive a comprehensive analysis including match scoring, skill gaps, optimization suggestions, 
            and interview preparation.
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                <div style="color: #64748b; font-size: 0.9rem;">Precision Matching</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
                <div style="color: #64748b; font-size: 0.9rem;">Skill Intelligence</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📝</div>
                <div style="color: #64748b; font-size: 0.9rem;">Resume Optimization</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎤</div>
                <div style="color: #64748b; font-size: 0.9rem;">Interview Prep</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style="text-align: center; padding: 2rem; margin-top: 3rem; border-top: 1px solid {Theme.BG_TERTIARY}; color: {Theme.TEXT_SECONDARY};">
    <p style="font-size: 0.9rem;">
        🎓 CareerSync AI 2026 | Empowering the Next Generation of Talent
    </p>
    <p style="font-size: 0.8rem; opacity: 0.7;">
        Remember: AI augments human potential. Always verify insights and trust your instincts.
    </p>
</div>
""", unsafe_allow_html=True)
