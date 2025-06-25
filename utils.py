import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os
import streamlit as st


def extract_name(email):
    """Extract name from email address and return the first name"""
    if not email:
        return ""

    # Extract the part before @ symbol
    name_part = email.split("@")[0]

    # Replace dots, underscores, hyphens with spaces
    name_part = re.sub(r"[._-]", " ", name_part)

    # Capitalize each word
    name_part = " ".join(word.capitalize() for word in name_part.split())

    # Get first name if possible
    name_parts = name_part.split()
    if name_parts:
        return name_parts[0]  # Return first name only

    return name_part  # Return full cleaned name if splitting doesn't work


def send_email(
    recipient_email,
    subject,
    greeting,
    body,
    signature,
    attachment=None,
    sender_email=None,
    sender_password=None,
    smtp_server=None,
    smtp_port=587,
):
    """Send an email with optional attachment"""
    # Get email credentials from session state or environment variables
    sender_email = sender_email or os.environ.get("EMAIL_ADDRESS")
    sender_password = sender_password or os.environ.get("EMAIL_PASSWORD")
    smtp_server = smtp_server or os.environ.get("SMTP_SERVER")
    smtp_port = int(smtp_port or os.environ.get("SMTP_PORT", 587))

    if not sender_email or not sender_password or not smtp_server:
        return False, "Email settings not configured. Please check the sidebar."

    try:
        # Create message
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject

        # Combine email parts
        email_text = f"{greeting}\n\n{body}\n\n{signature}"
        msg.attach(MIMEText(email_text, "plain"))

        # Add attachment if provided
        if attachment:
            attachment_part = MIMEApplication(
                attachment.getvalue(), Name=attachment.name
            )
            attachment_part["Content-Disposition"] = (
                f'attachment; filename="{attachment.name}"'
            )
            msg.attach(attachment_part)

        # Connect to server and send
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()

        return True, "Email sent successfully!"

    except Exception as e:
        return False, f"Error sending email: {str(e)}"


def get_default_templates():
    """Get default templates for greeting, body, and signature."""
    default_greeting = "Greetings Mr {name},"

    default_body = """I am a Software Engineer and an AI Expert with 6 years of experience based in Dubai.

I have been leading the development of Real Estate CRM solutions that's being used by 200+ agents in our company. I have a solid experience in delivering end to end solutions and a strong background on lots of 

I am ready to provide you with any details regarding my career or education, and I will be available to join you after 30 days if you consider my application for the {position} position."""

    default_signature = "Best wishes,\nOussama"

    return default_greeting, default_body, default_signature


def infer_smtp_settings(email_address):
    """Infer SMTP settings from email address"""
    if not email_address or "@" not in email_address:
        return None, None

    domain = email_address.split("@")[-1].lower()

    # Common email providers
    smtp_settings = {
        "gmail.com": ("smtp.gmail.com", 587),
        "outlook.com": ("smtp.office365.com", 587),
        "hotmail.com": ("smtp.office365.com", 587),
        "live.com": ("smtp.office365.com", 587),
        "yahoo.com": ("smtp.mail.yahoo.com", 587),
        "aol.com": ("smtp.aol.com", 587),
        "icloud.com": ("smtp.mail.me.com", 587),
        "me.com": ("smtp.mail.me.com", 587),
        "protonmail.com": ("smtp.protonmail.ch", 587),
        "zoho.com": ("smtp.zoho.com", 587),
        "yandex.com": ("smtp.yandex.com", 587),
        "mail.com": ("smtp.mail.com", 587),
        "gmx.com": ("smtp.gmx.com", 587),
    }

    # Try to find a match
    if domain in smtp_settings:
        return smtp_settings[domain]
    else:
        # Try generic approach
        return f"mail.{domain}", 587
