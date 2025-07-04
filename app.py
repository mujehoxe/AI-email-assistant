import streamlit as st
import os
from dotenv import load_dotenv
from template_manager import (
    init_template_state,
    render_template_sidebar,
    load_templates,
    get_template,
)
from ui import render_email_form
import threading
import time

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AI Email Assistant",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/email-sender-for-recruiters",
        "Report a bug": "https://github.com/yourusername/email-sender-for-recruiters/issues",
        "About": "Email Sender for Recruiters - A Streamlit app to send customized emails to recruiters with AI-powered template generation.",
    },
)

# Initialize session state
if "current_template" not in st.session_state:
    st.session_state.current_template = {
        "greeting": "Dear {name},",
        "body": "I am writing to express my interest in the {position} position at {company}. I believe my experience and skills align well with the requirements of this role.\n\nI would welcome the opportunity to discuss how my background would be a good fit for your team.\n\nThank you for considering my application.",
        "signature": "Best regards,\n[Your Name]",
    }

# Initialize resume storage in session state
if "resume_content" not in st.session_state:
    st.session_state.resume_content = ""
if "resume_filename" not in st.session_state:
    st.session_state.resume_filename = ""
if "customized_resume" not in st.session_state:
    st.session_state.customized_resume = ""
if "use_customized_resume" not in st.session_state:
    st.session_state.use_customized_resume = False
if "customized_resume_docx" not in st.session_state:
    st.session_state.customized_resume_docx = None
if "customized_resume_filename" not in st.session_state:
    st.session_state.customized_resume_filename = ""

# Initialize other session state variables
if "recipient_email" not in st.session_state:
    st.session_state.recipient_email = ""
if "recipient_name" not in st.session_state:
    st.session_state.recipient_name = ""
if "company_name" not in st.session_state:
    st.session_state.company_name = ""
if "position" not in st.session_state:
    st.session_state.position = ""
if "subject" not in st.session_state:
    st.session_state.subject = "Job Application"

# App title
st.title("📧 AI Email Assistant")
st.markdown(
    "Create professional job application emails with AI assistance. Upload your resume and job description to generate personalized templates."
)

# Initialize template state
init_template_state()

# Render sidebar
render_template_sidebar()

# Load templates from file
templates = load_templates()

# Add template selection to sidebar
st.sidebar.header("Templates")
template_names = list(templates.keys())
if template_names:
    selected_template = st.sidebar.selectbox(
        "Choose a template",
        options=template_names,
        index=0 if template_names else None,
    )

    if st.sidebar.button("Load Template"):
        if selected_template:
            template = get_template(selected_template)
            if template:
                st.session_state.current_template = template
                st.sidebar.success(f"Template '{selected_template}' loaded!")
                st.rerun()

# Render main form
render_email_form()
