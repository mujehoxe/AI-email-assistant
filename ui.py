import streamlit as st
from utils import extract_name, send_email
from template_manager import save_template
from ai_helper import (
    generate_improved_template,
    start_g4f_api_server,
    cancel_ai_request,
)
import io
import traceback
import threading

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


def render_email_form():
    """Render the email form"""
    # Store form values in session state if not already there
    if "recipient_email" not in st.session_state:
        st.session_state.recipient_email = ""
    if "employer_name" not in st.session_state:
        st.session_state.employer_name = ""
    if "position" not in st.session_state:
        st.session_state.position = ""
    if "subject" not in st.session_state:
        st.session_state.subject = "Job Application"
    if "ai_request_active" not in st.session_state:
        st.session_state.ai_request_active = False

    col1, col2 = st.columns([1, 1])

    # Email configuration column
    recipient_email, employer_name, position, subject, resume_file = (
        render_email_config(col1)
    )

    # Email template column
    greeting, email_body, signature = render_email_template(
        col2, employer_name, position, resume_file
    )

    # Send email button
    st.divider()
    if st.button("Send Email", type="primary"):
        if not recipient_email:
            st.error("Please enter a recipient email address.")
        else:
            success, message = send_email(
                recipient_email=recipient_email,
                subject=subject,
                greeting=greeting,
                body=email_body,
                signature=signature,
                attachment=resume_file,
            )

            if success:
                st.success(message)
            else:
                st.error(message)


def render_email_config(column):
    """Render the email configuration section"""
    with column:
        st.subheader("Email Configuration")

        recipient_email = st.text_input(
            "Recipient Email",
            value=st.session_state.recipient_email,
            help="Enter the recruiter's email address",
        )
        st.session_state.recipient_email = recipient_email

        # Add employer name field
        employer_name = st.text_input(
            "Employer Name",
            value=st.session_state.employer_name,
            help="Enter the employer's name (if left empty, will use email)",
        )
        st.session_state.employer_name = employer_name

        # If employer name is empty and we have an email, extract from email
        if not employer_name and recipient_email:
            employer_name = extract_name(recipient_email)

        # Add position/designation field
        position = st.text_input(
            "Position/Designation",
            value=st.session_state.position,
            help="Enter the position you're applying for",
        )
        st.session_state.position = position

        # Update subject with position if available
        if position and not st.session_state.subject_manually_set:
            subject_value = f"Application for the {position} position"
        else:
            subject_value = st.session_state.subject

        subject = st.text_input("Subject", value=subject_value)

        # Track if subject was manually set
        if "subject_manually_set" not in st.session_state:
            st.session_state.subject_manually_set = False
        if subject != subject_value:
            st.session_state.subject_manually_set = True

        st.session_state.subject = subject

        # File uploader for resume
        resume_file = st.file_uploader(
            "Upload Resume (PDF recommended)", type=["pdf", "docx", "doc"]
        )

        if resume_file:
            st.success(f"Resume uploaded: {resume_file.name}")
            if resume_parsing_available:
                with st.expander("Resume Preview"):
                    resume_text = extract_text_from_resume(resume_file)
                    if resume_text:
                        st.text_area(
                            "Extracted Text",
                            value=(
                                resume_text[:1000] + "..."
                                if len(resume_text) > 1000
                                else resume_text
                            ),
                            height=150,
                        )
                    else:
                        st.warning("Could not extract text from resume.")

        return recipient_email, employer_name, position, subject, resume_file


def ai_template_generation_thread(
    job_description, resume_text, current_template, placeholder
):
    """Run AI template generation in a separate thread"""
    try:
        # Set the active flag
        st.session_state.ai_request_active = True

        # Generate improved template
        result = generate_improved_template(
            job_description=job_description,
            resume_content=resume_text,
            current_template=current_template,
        )

        # Clear the active flag
        st.session_state.ai_request_active = False

        # Process the result
        if result["success"]:
            # Update template values in session state for the next rerun
            st.session_state.current_template_update = {
                "greeting": result["template"]["greeting"],
                "body": result["template"]["body"],
                "signature": result["template"]["signature"],
            }

            # Update form fields with AI suggestions
            if "position" in result["template"] and result["template"]["position"]:
                st.session_state.position = result["template"]["position"]

            if "employer" in result["template"] and result["template"]["employer"]:
                st.session_state.employer_name = result["template"]["employer"]

            if "subject" in result["template"] and result["template"]["subject"]:
                st.session_state.subject = result["template"]["subject"]
                st.session_state.subject_manually_set = True

            if (
                "recipient_email" in result["template"]
                and result["template"]["recipient_email"]
            ):
                st.session_state.recipient_email = result["template"]["recipient_email"]

            # Save as a new template
            if st.session_state.position:
                template_name = f"AI Template - {st.session_state.position}"
            else:
                template_name = "AI Generated Template"

            save_template(
                template_name,
                result["template"]["greeting"],
                result["template"]["body"],
                result["template"]["signature"],
            )

            # Update status for next rerun
            st.session_state.ai_result_status = "success"
            st.session_state.ai_result_message = f"AI template generated and saved as '{template_name}'! Form fields have been updated."
        else:
            # Update status for next rerun
            st.session_state.ai_result_status = "error"
            st.session_state.ai_result_message = (
                f"Failed to generate template: {result.get('error', 'Unknown error')}"
            )

        # Trigger a rerun to update the UI
        placeholder.empty()
        st.experimental_rerun()

    except Exception as e:
        # Clear the active flag
        st.session_state.ai_request_active = False

        # Update status for next rerun
        st.session_state.ai_result_status = "error"
        st.session_state.ai_result_message = f"Error generating template: {str(e)}"

        # Log the error
        st.error(f"Error generating template: {str(e)}")
        st.error(traceback.format_exc())

        # Trigger a rerun to update the UI
        placeholder.empty()
        st.experimental_rerun()


def render_email_template(column, employer_name, position, resume_file=None):
    """Render the email template section"""
    with column:
        st.subheader("Email Template")

        # Get current template values
        current_greeting = st.session_state.current_template.get("greeting", "")
        current_body = st.session_state.current_template.get("body", "")
        current_signature = st.session_state.current_template.get("signature", "")

        # Update template if we have new values from AI
        if hasattr(st.session_state, "current_template_update"):
            current_greeting = st.session_state.current_template_update.get(
                "greeting", current_greeting
            )
            current_body = st.session_state.current_template_update.get(
                "body", current_body
            )
            current_signature = st.session_state.current_template_update.get(
                "signature", current_signature
            )
            # Clear the update to avoid applying it again
            del st.session_state.current_template_update

        # AI Template Generation
        st.info(
            "💡 Use AI to generate a tailored template based on job description and your resume"
        )

        with st.expander("AI Template Generator"):
            # Check G4F API server status
            if st.button("Check/Start G4F API Server"):
                try:
                    start_g4f_api_server()
                    st.success("G4F API server is running")
                except Exception as e:
                    st.error(f"Error starting G4F API server: {str(e)}")
                    st.error(traceback.format_exc())

            job_description = st.text_area(
                "Job Description",
                placeholder="Paste the job description here to generate a tailored template",
                height=150,
            )

            # Create a placeholder for the spinner and status messages
            placeholder = st.empty()

            # Show result status from previous run if available
            if hasattr(st.session_state, "ai_result_status"):
                if st.session_state.ai_result_status == "success":
                    st.success(st.session_state.ai_result_message)
                else:
                    st.error(st.session_state.ai_result_message)
                # Clear the status to avoid showing it again
                del st.session_state.ai_result_status
                del st.session_state.ai_result_message

            # Display generate/cancel buttons based on active state
            col1, col2 = st.columns([1, 1])

            with col1:
                generate_button = st.button(
                    "Generate AI Template", disabled=st.session_state.ai_request_active
                )

            with col2:
                if st.session_state.ai_request_active:
                    if st.button("Cancel Generation"):
                        cancel_ai_request()
                        st.session_state.ai_request_active = False
                        st.warning("AI request canceled")
                        st.experimental_rerun()

            if generate_button:
                if not job_description:
                    st.warning(
                        "Please provide a job description to generate a template"
                    )
                else:
                    with placeholder.container():
                        st.info("Starting AI template generation...")
                        st.spinner("Generating template with AI...")

                        # Extract text from resume if available
                        resume_text = (
                            extract_text_from_resume(resume_file) if resume_file else ""
                        )

                        if not resume_text and resume_file:
                            st.warning(
                                "Could not extract text from your resume. The template will be generated based only on the job description."
                            )

                        # Current template data
                        current_template = {
                            "greeting": current_greeting,
                            "body": current_body,
                            "signature": current_signature,
                        }

                        # Start generation in a separate thread
                        generation_thread = threading.Thread(
                            target=ai_template_generation_thread,
                            args=(
                                job_description,
                                resume_text,
                                current_template,
                                placeholder,
                            ),
                        )
                        generation_thread.daemon = True
                        generation_thread.start()

        # Replace {name} placeholder with the employer name
        greeting_with_name = current_greeting.replace(
            "{name}", employer_name if employer_name else ""
        )

        # Replace {position} in body if it exists
        body_with_position = current_body
        if position:
            body_with_position = current_body.replace("{position}", position)

        # Separate fields for greeting, body and signature
        greeting = st.text_input("Greeting", value=greeting_with_name)
        email_body = st.text_area("Email Body", value=body_with_position, height=250)
        signature = st.text_area("Signature", value=current_signature, height=80)

        # Template saving section
        st.divider()
        col_save1, col_save2 = st.columns([3, 1])
        with col_save1:
            template_name = st.text_input(
                "Template Name", placeholder="Enter a name to save this template"
            )
        with col_save2:
            if st.button("Save Template") and template_name:
                save_template(template_name, greeting, email_body, signature)
                st.success(f"Template '{template_name}' saved!")

        return greeting, email_body, signature
