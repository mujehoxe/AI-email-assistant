import streamlit as st
import os
from dotenv import load_dotenv
from template_manager import init_template_state, render_template_sidebar
from ui import render_email_form
from settings import render_settings_sidebar
from utils import init_user_settings
import threading
import time

# Load environment variables
load_dotenv()

# App title and configuration
st.set_page_config(
    page_title="Email Sender for Recruiters",
    layout="wide",
    menu_items={
        "Get Help": "https://github.com/yourusername/email-sender-for-recruiters",
        "Report a bug": "https://github.com/yourusername/email-sender-for-recruiters/issues",
        "About": "Email Sender for Recruiters - A Streamlit app to send customized emails to recruiters with AI-powered template generation.",
    },
)

# Initialize user settings
init_user_settings()

# Main title
st.title("Email Sender for Recruiters")

# Check if settings have been saved
if not st.session_state.settings_saved:
    st.warning(
        "Please configure your email settings in the sidebar before sending emails."
    )

# Initialize template state
init_template_state()

# Render sidebar components
render_template_sidebar()
render_settings_sidebar()

# Render main form
render_email_form()

# Footer
st.markdown("---")
st.markdown(f"Created by {st.session_state.user_name}")
