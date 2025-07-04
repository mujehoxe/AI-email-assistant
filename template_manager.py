import streamlit as st
import json
import os
from utils import get_default_templates

TEMPLATES_FILE = "templates.json"


def init_template_state():
    """Initialize template state in session state"""
    # Get default templates
    default_greeting, default_body, default_signature = get_default_templates()

    # Initialize templates dictionary if not exists
    if "templates" not in st.session_state:
        st.session_state.templates = {}

    # Add a default template if no templates exist
    if not st.session_state.templates and "Default" not in st.session_state.templates:
        st.session_state.templates["Default"] = {
            "greeting": default_greeting,
            "body": default_body,
            "signature": default_signature,
        }

    # Initialize current template
    if "current_template" not in st.session_state:
        # Try to load last used template
        if "last_used_template" in st.session_state:
            st.session_state.current_template = st.session_state.last_used_template
        else:
            # Use default template
            st.session_state.current_template = {
                "greeting": default_greeting,
                "body": default_body,
                "signature": default_signature,
            }
            # Save as last used template
            st.session_state.last_used_template = st.session_state.current_template


def save_template(name, greeting, body, signature):
    """Save a template to the templates file"""
    templates = load_templates()

    # Add or update template
    templates[name] = {"greeting": greeting, "body": body, "signature": signature}

    try:
        with open(TEMPLATES_FILE, "w") as f:
            json.dump(templates, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving template: {str(e)}")
        return False


def delete_template(name):
    """Delete a template from the templates file"""
    templates = load_templates()

    if name in templates:
        del templates[name]

        try:
            with open(TEMPLATES_FILE, "w") as f:
                json.dump(templates, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error deleting template: {str(e)}")

    return False


def load_template(name):
    """Load template from session state"""
    if name in st.session_state.templates:
        template = st.session_state.templates[name]
        st.session_state.current_template = template

        # Save as last used template
        st.session_state.last_used_template = template

        return True
    return False


def load_templates():
    """Load templates from JSON file"""
    if os.path.exists(TEMPLATES_FILE):
        try:
            with open(TEMPLATES_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading templates: {str(e)}")
    return {}


def get_template(name):
    """Get a specific template by name"""
    templates = load_templates()
    return templates.get(name, None)


def render_template_sidebar():
    """Render the template sidebar"""
    # This function is kept for compatibility but template selection
    # is now handled in app.py to avoid duplication
    pass
