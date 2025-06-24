import streamlit as st
import json
import os
from utils import get_default_templates


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
    """Save template to session state"""
    st.session_state.templates[name] = {
        "greeting": greeting,
        "body": body,
        "signature": signature,
    }

    # Save as last used template
    st.session_state.last_used_template = {
        "greeting": greeting,
        "body": body,
        "signature": signature,
    }


def delete_template(name):
    """Delete template from session state"""
    if name in st.session_state.templates:
        del st.session_state.templates[name]


def load_template(name):
    """Load template from session state"""
    if name in st.session_state.templates:
        template = st.session_state.templates[name]
        st.session_state.current_template = template

        # Save as last used template
        st.session_state.last_used_template = template

        return True
    return False


def render_template_sidebar():
    """Render the template sidebar"""
    st.sidebar.header("Template Library")

    # Template selection in sidebar
    if st.session_state.templates:
        selected_template = st.sidebar.selectbox(
            "Select a template", options=list(st.session_state.templates.keys())
        )

        if selected_template:
            # Auto-load template when selected
            if st.sidebar.button("Load Selected Template"):
                if load_template(selected_template):
                    st.sidebar.success(f"Template '{selected_template}' loaded!")
                    st.experimental_rerun()

            # Delete button
            if st.sidebar.button("Delete Selected Template"):
                delete_template(selected_template)
                st.sidebar.warning(f"Template '{selected_template}' deleted!")
                st.experimental_rerun()
    else:
        st.sidebar.info("No saved templates yet")
