import streamlit as st
from utils import extract_name, send_email
from template_manager import save_template
from ai_helper import (
    generate_improved_template,
    start_g4f_api_server,
    cancel_ai_request,
    is_request_in_progress,
    AVAILABLE_MODELS,
    get_available_models,
    check_api_server_status,
    extract_contact_info,
)
import io
import traceback
import os
import re
import json

# Try importing PDF and docx libraries
try:
    import PyPDF2
    import docx

    resume_parsing_available = True
except ImportError:
    st.warning(
        "Resume parsing libraries not available. Install PyPDF2 and python-docx for resume parsing."
    )
    resume_parsing_available = False


def extract_text_from_resume(resume_file):
    """Extract text content from resume file"""
    if resume_file is None or not resume_parsing_available:
        return ""

    try:
        file_extension = resume_file.name.split(".")[-1].lower()

        if file_extension == "pdf":
            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(resume_file.getvalue()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text

        elif file_extension in ["docx", "doc"]:
            # Extract text from Word document
            doc = docx.Document(io.BytesIO(resume_file.getvalue()))
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text

        else:
            # For other formats, return empty string
            st.warning(
                f"Unsupported file format: {file_extension}. Only PDF and DOCX are supported."
            )
            return ""

    except Exception as e:
        st.error(f"Error extracting text from resume: {str(e)}")
        st.error(traceback.format_exc())
        return ""


def infer_smtp_server(email):
    """Infer SMTP server from email address"""
    if not email:
        return ""

    domain = email.split("@")[-1].lower()

    # Common email providers
    smtp_servers = {
        "gmail.com": "smtp.gmail.com",
        "outlook.com": "smtp.office365.com",
        "hotmail.com": "smtp.office365.com",
        "yahoo.com": "smtp.mail.yahoo.com",
        "aol.com": "smtp.aol.com",
        "icloud.com": "smtp.mail.me.com",
        "protonmail.com": "smtp.protonmail.ch",
    }

    return smtp_servers.get(domain, f"mail.{domain}")


def render_email_settings_sidebar():
    """Render email settings in sidebar"""
    st.sidebar.header("Email Settings")

    # Get environment variables if they exist
    default_email = os.environ.get("EMAIL_ADDRESS", "")
    default_password = os.environ.get("EMAIL_PASSWORD", "")

    # Email credentials
    email = st.sidebar.text_input("Your Email Address", value=default_email)
    if email:
        # Store in session state
        st.session_state.sender_email = email

    password = st.sidebar.text_input(
        "Email Password", type="password", value=default_password
    )
    if password:
        # Store in session state
        st.session_state.sender_password = password

    # SMTP server settings
    inferred_server = infer_smtp_server(email) if email else ""
    smtp_server = st.sidebar.text_input("SMTP Server", value=inferred_server)
    if smtp_server:
        st.session_state.smtp_server = smtp_server

    smtp_port = st.sidebar.selectbox(
        "SMTP Port",
        options=[587, 465, 25],
        index=0,
        help="Port 587 uses STARTTLS, 465 uses SSL/TLS",
    )
    st.session_state.smtp_port = smtp_port

    # Save settings to .env file
    if st.sidebar.button("Save Email Settings"):
        try:
            with open(".env", "w") as f:
                f.write(f"EMAIL_ADDRESS={email}\n")
                f.write(f"EMAIL_PASSWORD={password}\n")
                f.write(f"SMTP_SERVER={smtp_server}\n")
                f.write(f"SMTP_PORT={smtp_port}\n")
            st.sidebar.success("Email settings saved to .env file")
        except Exception as e:
            st.sidebar.error(f"Error saving settings: {str(e)}")

    # Add G4F API server controls to sidebar
    st.sidebar.header("AI Server Settings")

    # Check if API server is running
    server_status = check_api_server_status()
    status_color = "🟢" if server_status else "🔴"
    st.sidebar.write(
        f"{status_color} Server status: {'Running' if server_status else 'Stopped'}"
    )

    if st.sidebar.button("Start G4F API Server"):
        try:
            start_g4f_api_server()
            st.sidebar.success("G4F API server is running")
            # Force fetch the models after starting the server
            try:
                models = get_available_models()
                if models and models != AVAILABLE_MODELS:
                    st.session_state.ai_models = models
                    st.sidebar.success(f"Found {len(models)} models")
                    st.experimental_rerun()
            except Exception as e:
                st.sidebar.error(f"Error fetching models: {str(e)}")
        except Exception as e:
            st.sidebar.error(f"Error starting G4F API server: {str(e)}")
            st.sidebar.error(traceback.format_exc())

    if st.sidebar.button("Refresh AI Models"):
        try:
            models = get_available_models()
            st.session_state.ai_models = models
            st.sidebar.success(f"Found {len(models)} models")
            st.experimental_rerun()
        except Exception as e:
            st.sidebar.error(f"Error refreshing models: {str(e)}")


def render_email_form():
    """Render the email form with template generation and sending functionality"""
    from ai_helper import (
        get_available_models,
        generate_improved_template,
        check_api_server_status,
        is_request_in_progress,
        cancel_ai_request,
        extract_contact_info,
    )
    from template_manager import save_template
    from utils import send_email, infer_smtp_settings

    # Create a two-column layout
    col1, col2 = st.columns([3, 2])

    with col1:
        # Email Form
        st.header("Email Generator")

        # File upload and processing
        resume_file = st.file_uploader(
            "Upload your resume (PDF or DOCX)", type=["pdf", "docx"]
        )
        resume_content = ""

        if resume_file:
            try:
                if resume_file.name.endswith(".pdf"):
                    from PyPDF2 import PdfReader

                    pdf = PdfReader(resume_file)
                    resume_content = ""
                    for page in pdf.pages:
                        resume_content += page.extract_text()
                elif resume_file.name.endswith(".docx"):
                    from docx import Document

                    doc = Document(resume_file)
                    resume_content = "\n".join([para.text for para in doc.paragraphs])

                st.success(f"Resume processed: {resume_file.name}")
            except Exception as e:
                st.error(f"Error processing resume: {str(e)}")

        # Job description input
        st.markdown("### Job Description")
        job_description = st.text_area(
            "Paste the job description here",
            height=200,
            key="job_description",
            help="Copy and paste the entire job posting here",
        )

        # Extract contact info from job description
        contact_info = {}
        if job_description:
            contact_info = extract_contact_info(job_description)

            # Update session state with extracted info
            if contact_info.get("name"):
                st.session_state.recipient_name = contact_info["name"]
            if contact_info.get("email"):
                st.session_state.recipient_email = contact_info["email"]
            if contact_info.get("company"):
                st.session_state.company_name = contact_info["company"]

        # Template generator
        st.markdown("### AI Template Generator")

        # Check API server status
        server_status = "Not Running"
        if check_api_server_status():
            server_status = "Running"

        st.markdown(f"API Server Status: **{server_status}**")

        # Get available models
        if "ai_models" not in st.session_state:
            st.session_state.ai_models = get_available_models()

        model_count = len(st.session_state.ai_models)
        st.markdown(f"Available Models: **{model_count}**")

        # Model selection
        selected_model = st.selectbox(
            "Select AI Model",
            options=st.session_state.ai_models,
            index=0 if st.session_state.ai_models else None,
        )

        # Refresh models button
        if st.button("Refresh Models List"):
            st.session_state.ai_models = get_available_models()
            st.experimental_rerun()

        # Generate template button
        generate_col, cancel_col = st.columns([3, 1])

        with generate_col:
            if generate_button := st.button(
                "Generate Email Template",
                disabled=is_request_in_progress()
                or not job_description
                or not resume_content,
            ):
                with st.spinner("Generating template..."):
                    # Get current template for reference
                    current_template = st.session_state.current_template

                    # Generate template
                    result = generate_improved_template(
                        job_description,
                        resume_content,
                        current_template,
                        model=selected_model,
                    )

                    # Store result in session state
                    st.session_state.result = result

                    if result.get("success", False):
                        template = result["template"]

                        # Update template in session state
                        st.session_state.current_template = {
                            "greeting": template.get("greeting", ""),
                            "body": template.get("body", ""),
                            "signature": template.get("signature", ""),
                        }

                        # Update other extracted information
                        if template.get("position"):
                            st.session_state.position = template["position"]
                        if template.get("employer"):
                            st.session_state.company_name = template["employer"]
                        if template.get("subject"):
                            st.session_state.subject = template["subject"]
                        if template.get("recipient_email"):
                            st.session_state.recipient_email = template[
                                "recipient_email"
                            ]

                        st.success("Template generated successfully!")
                        st.experimental_rerun()
                    else:
                        st.error(
                            f"Failed to generate template: {result.get('error', 'Unknown error')}"
                        )

        with cancel_col:
            if st.button("Cancel", disabled=not is_request_in_progress()):
                if cancel_ai_request():
                    st.warning("Request cancelled")
                    st.experimental_rerun()

    with col2:
        # Email preview with placeholders filled in
        st.header("Email Preview")

        # Get values for placeholders
        recipient_name = st.session_state.get("recipient_name", "")
        company_name = st.session_state.get("company_name", "")
        position_name = st.session_state.get("position", "")

        # If we don't have recipient name but have email, extract from email
        if not recipient_name and st.session_state.get("recipient_email"):
            from utils import extract_name

            recipient_name = extract_name(st.session_state.get("recipient_email"))

        # Replace placeholders in template
        greeting = st.session_state.current_template["greeting"]
        if "{name}" in greeting:
            greeting = greeting.replace("{name}", recipient_name or "Hiring Manager")

        body = st.session_state.current_template["body"]
        if "{position}" in body:
            body = body.replace("{position}", position_name or "position")
        if "{company}" in body:
            body = body.replace("{company}", company_name or "the company")

        signature = st.session_state.current_template["signature"]

        # Display editable preview
        st.markdown("### Greeting")
        preview_greeting = st.text_area(
            "Edit greeting",
            greeting,
            height=50,
            key="preview_greeting",
        )

        st.markdown("### Body")
        preview_body = st.text_area(
            "Edit body",
            body,
            height=200,
            key="preview_body",
        )

        st.markdown("### Signature")
        preview_signature = st.text_area(
            "Edit signature",
            signature,
            height=100,
            key="preview_signature",
        )

        # Copy to clipboard button
        if st.button("Copy Template to Clipboard"):
            full_email = f"{preview_greeting}\n\n{preview_body}\n\n{preview_signature}"
            # Use JavaScript to copy to clipboard
            js_code = f"""
            <script>
                const text = `{full_email.replace('"', '\\"').replace('\n', '\\n')}`;
                const el = document.createElement('textarea');
                el.value = text;
                document.body.appendChild(el);
                el.select();
                document.execCommand('copy');
                document.body.removeChild(el);
                alert('Email template copied to clipboard!');
            </script>
            """
            st.components.v1.html(js_code, height=0)
            st.success("Email template copied to clipboard!")

        # Template save form
        st.markdown("### Save Template")
        template_name = st.text_input("Template Name", "")

        if st.button("Save Template", disabled=not template_name):
            if save_template(
                template_name, preview_greeting, preview_body, preview_signature
            ):
                st.success(f"Template '{template_name}' saved!")
            else:
                st.error("Failed to save template")
