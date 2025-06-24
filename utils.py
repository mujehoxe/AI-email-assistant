import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os
import streamlit as st


def extract_name(email):
    """Extract name from the first part of the email address."""
    if not email:
        return ""

    # Extract the part before @ and before any dots or numbers
    match = re.match(r"^([a-zA-Z]+)", email.split("@")[0])
    if match:
        name = match.group(1)
        # Capitalize the first letter
        return name.capitalize()
    return ""


def infer_smtp_settings(email):
    """Infer SMTP settings from email domain."""
    if not email or "@" not in email:
        return None, None

    # Extract domain from email
    domain = email.split("@")[1].lower()

    # Common email providers SMTP settings
    smtp_settings = {
        "gmail.com": ("smtp.gmail.com", 587),
        "outlook.com": ("smtp-mail.outlook.com", 587),
        "hotmail.com": ("smtp-mail.outlook.com", 587),
        "live.com": ("smtp-mail.outlook.com", 587),
        "yahoo.com": ("smtp.mail.yahoo.com", 587),
        "yahoo.co.uk": ("smtp.mail.yahoo.com", 587),
        "yahoo.co.in": ("smtp.mail.yahoo.com", 587),
        "aol.com": ("smtp.aol.com", 587),
        "zoho.com": ("smtp.zoho.com", 587),
        "protonmail.com": ("smtp.protonmail.ch", 587),
        "icloud.com": ("smtp.mail.me.com", 587),
        "mail.com": ("smtp.mail.com", 587),
        "gmx.com": ("smtp.gmx.com", 587),
        "yandex.com": ("smtp.yandex.com", 587),
    }

    # Check if domain is in our known list
    if domain in smtp_settings:
        return smtp_settings[domain]

    # Try to guess based on common patterns
    # For example, many companies use smtp.domain.com
    try:
        # Try to connect to smtp.domain.com
        test_server = f"mail.{domain}"
        with smtplib.SMTP(test_server, 587, timeout=2) as server:
            server.ehlo()
            return test_server, 587
    except:
        pass

    # If we couldn't determine the settings, return None
    return None, None


def send_email(recipient_email, subject, greeting, body, signature, attachment=None):
    """Send email using SMTP."""
    try:
        # Get email credentials from session state or environment variables
        if "user_email" in st.session_state and st.session_state.user_email:
            sender_email = st.session_state.user_email
        else:
            sender_email = os.getenv("EMAIL_ADDRESS")

        if "user_password" in st.session_state and st.session_state.user_password:
            password = st.session_state.user_password
        else:
            password = os.getenv("EMAIL_PASSWORD")

        # Get SMTP settings
        smtp_server = st.session_state.smtp_server
        smtp_port = st.session_state.smtp_port

        # Create a multipart message
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject

        # Combine greeting, body and signature
        full_body = f"{greeting}\n\n{body}\n\n{signature}"

        # Add body to email
        msg.attach(MIMEText(full_body, "plain"))

        # Add attachment if provided
        if attachment is not None:
            attachment_name = os.path.basename(attachment.name)
            part = MIMEApplication(attachment.getbuffer(), Name=attachment_name)
            part["Content-Disposition"] = f'attachment; filename="{attachment_name}"'
            msg.attach(part)

        # Create SMTP session
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)

        return True, "Email sent successfully!"

    except Exception as e:
        return False, f"Error: {str(e)}"


def get_default_templates():
    """Get default templates for greeting, body, and signature."""
    default_greeting = "Greetings Mr {name},"

    default_body = """I am a Software Engineer and an AI Expert with 6 years of experience based in Dubai.

I have been leading the development of Real Estate CRM solutions that's being used by 200+ agents in our company. I have a solid experience in delivering end to end solutions and a strong background on lots of 

I am ready to provide you with any details regarding my career or education, and I will be available to join you after 30 days if you consider my application for the {position} position."""

    # Use user's name from session state if available
    if "user_name" in st.session_state and st.session_state.user_name:
        default_signature = f"Best wishes,\n{st.session_state.user_name}"
    else:
        default_signature = "Best wishes"

    return default_greeting, default_body, default_signature


def init_user_settings():
    """Initialize user settings in session state"""
    if "user_email" not in st.session_state:
        st.session_state.user_email = os.getenv("EMAIL_ADDRESS", "")

    if "user_password" not in st.session_state:
        st.session_state.user_password = os.getenv("EMAIL_PASSWORD", "")

    if "user_name" not in st.session_state:
        st.session_state.user_name = extract_name(st.session_state.user_email)

    if "smtp_server" not in st.session_state:
        st.session_state.smtp_server = infer_smtp_settings(st.session_state.user_email)[
            0
        ]

    if "smtp_port" not in st.session_state:
        st.session_state.smtp_port = infer_smtp_settings(st.session_state.user_email)[1]

    if "settings_saved" not in st.session_state:
        st.session_state.settings_saved = False
