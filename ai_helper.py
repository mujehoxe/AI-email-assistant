import streamlit as st
from openai import OpenAI
import json
import subprocess
import time
import threading
import os
import traceback
import re
import requests

# Get G4F API endpoint from environment variables or use default
G4F_API_HOST = os.environ.get("G4F_API_HOST", "localhost")
G4F_API_PORT = os.environ.get("G4F_API_PORT", "1337")
G4F_API_BASE_URL = f"http://{G4F_API_HOST}:{G4F_API_PORT}/v1"

# Global flag to check if API server is running
api_server_running = False
# Global flag to check if a request is in progress
request_in_progress = False
# Global flag to cancel a request
cancel_request = False

# Fallback models in case we can't fetch from API
FALLBACK_MODELS = [
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4o",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "gemini-pro",
    "gemini-1.5-pro-latest",
]


def get_available_models():
    """Get list of available models from G4F API"""
    try:
        # Try to get models from API
        response = requests.get(f"{G4F_API_BASE_URL}/models")
        if response.status_code == 200:
            models_data = response.json()

            # Debug - log the response
            # st.write("API Response:", models_data)

            # Extract model IDs - handle format from your curl output
            if "data" in models_data and isinstance(models_data["data"], list):
                available_models = []
                for model in models_data["data"]:
                    if "id" in model and model["id"] != "default":
                        # Skip the default model and add others
                        available_models.append(model["id"])

                # If we found models, update the session state and return them
                if available_models:
                    # Make sure we don't have duplicates
                    available_models = list(set(available_models))
                    # Sort models alphabetically for better UI
                    available_models.sort()
                    # Update session state if it exists
                    if "ai_models" in st.session_state:
                        st.session_state.ai_models = available_models
                    return available_models

            st.warning("API returned models but in unexpected format")
    except Exception as e:
        st.warning(f"Could not fetch available models from G4F API: {e}")
        # More detailed error for debugging
        # st.warning(f"Details: {traceback.format_exc()}")

    # Return fallback models if API request failed
    return FALLBACK_MODELS


# Initialize models list - will be updated when API server starts
AVAILABLE_MODELS = FALLBACK_MODELS


def extract_email_from_text(text):
    """Extract email addresses from text"""
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else ""


def extract_company_name(job_description):
    """Extract company name from job description"""
    company_name = ""

    # Common patterns for company names in job descriptions
    patterns = [
        r"(?:at|with|for|join)\s+([A-Z][A-Za-z0-9\s&\.,]+?)(?:\s+is|\s+as|\s+in|\s*,|\s*\.|\s+we|\s+our|\s+seeking|\s+hiring|\s+looking)",
        r"About\s+([A-Z][A-Za-z0-9\s&\.,]+?)(?:\s+is|\s+as|\s+in|\s*,|\s*\.|\s+we|\s+our)",
        r"([A-Z][A-Za-z0-9\s&\.,]+?)\s+is\s+(?:seeking|looking|hiring|searching|recruiting)",
        r"Welcome\s+to\s+([A-Z][A-Za-z0-9\s&\.,]+)",
        r"About\s+the\s+Company[:\s]+([A-Z][A-Za-z0-9\s&\.,]+)",
        r"Company:\s+([A-Z][A-Za-z0-9\s&\.,]+)",
    ]

    # Try each pattern
    for pattern in patterns:
        matches = re.search(pattern, job_description)
        if matches:
            company_name = matches.group(1).strip()
            # Clean up any trailing punctuation
            company_name = re.sub(r"[,\.\s]+$", "", company_name)
            break

    return company_name


def extract_contact_info(job_description):
    """Extract contact information from job description"""
    # Extract email
    email = extract_email_from_text(job_description)

    # Extract company name
    company_name = extract_company_name(job_description)

    # Try to extract name associated with the email
    name = ""

    # Look for common patterns like "Contact: John Doe (john.doe@example.com)"
    name_patterns = [
        (
            r"(?:contact|send|email|apply|resume to|cv to|application to)[:\s]+([\w\s]+)[\s\(<]+"
            + re.escape(email)
            if email
            else ""
        ),
        r"([\w\s]+)[\s\(<]+" + re.escape(email) if email else "",
        r"(?:contact|send|email|apply|resume to|cv to|application to)[:\s]+([\w\s]+)",
        r"(?:recruiter|hiring manager|hr manager|contact person)[:\s]+([\w\s]+)",
    ]

    # Try to find name in job description
    if email:
        for pattern in name_patterns:
            if pattern:  # Skip empty patterns
                matches = re.search(pattern, job_description, re.IGNORECASE)
                if matches:
                    name = matches.group(1).strip()
                    break

    # If name not found in job description but we have email, extract from email
    if not name and email:
        # Extract name from email address
        name = extract_name(email)

    # Ensure company name is not empty
    if not company_name:
        company_name = "the company"

    return {"email": email, "name": name, "company": company_name}


def start_g4f_api_server():
    """Start the G4F API server in a separate thread"""
    global api_server_running, AVAILABLE_MODELS

    if not api_server_running:

        def run_server():
            try:
                st.info("Starting G4F API server...")
                process = subprocess.Popen(
                    ["python", "-c", "from g4f.api import run_api; run_api()"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                # Log any output from the subprocess
                for line in process.stderr:
                    st.error(f"G4F API server error: {line.decode().strip()}")
                for line in process.stdout:
                    st.info(f"G4F API server: {line.decode().strip()}")

            except Exception as e:
                st.error(f"Failed to start G4F API server: {str(e)}")
                st.error(traceback.format_exc())

        # Start server in a separate thread
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True  # Thread will exit when main program exits
        server_thread.start()

        # Wait for server to start
        time.sleep(5)
        api_server_running = True

        # Add info to session state
        st.session_state.api_server_status = "Running"

        # Try to update available models
        try:
            # Wait a bit more for the server to be fully initialized
            time.sleep(2)
            fresh_models = get_available_models()
            if fresh_models and len(fresh_models) > 0:
                AVAILABLE_MODELS = fresh_models
                # Also update session state if it exists
                if "ai_models" in st.session_state:
                    st.session_state.ai_models = fresh_models
        except Exception as e:
            st.warning(f"Could not update model list: {e}")


def check_api_server_status():
    """Check if the G4F API server is running"""
    try:
        response = requests.get(f"{G4F_API_BASE_URL}/models")
        if response.status_code == 200:
            # Also try to update models when checking status
            try:
                fresh_models = get_available_models()
                if fresh_models and len(fresh_models) > 0:
                    if "ai_models" in st.session_state:
                        st.session_state.ai_models = fresh_models
            except:
                pass
            return True
    except:
        pass
    return False


def get_ai_client():
    """Get OpenAI client configured to use G4F API"""
    # Initialize the OpenAI client
    client = OpenAI(
        api_key="secret",  # Using "secret" as the API key since G4F doesn't require a real one
        base_url=G4F_API_BASE_URL,  # Point to the G4F API endpoint
    )
    return client


def generate_improved_template(
    job_description, resume_content, current_template, model="gpt-4o-mini"
):
    """Generate an improved template based on job description and resume"""
    global request_in_progress, cancel_request

    try:
        # Set request in progress flag
        request_in_progress = True
        cancel_request = False

        # Extract contact info from job description
        contact_info = extract_contact_info(job_description)

        # Add company name to log
        if contact_info["company"]:
            st.info(f"Detected company name: {contact_info['company']}")

        # Ensure API server is running
        if not check_api_server_status():
            start_g4f_api_server()

        # Get client
        client = get_ai_client()

        # Construct the prompt - note the double braces to escape placeholders
        prompt = f"""
        Create a personalized email template for a job application based on this information:
        
        ## Job Description:
        {job_description}
        
        ## My Resume:
        {resume_content}
        
        ## Current Email Template:
        Greeting: {current_template.get('greeting', '')}
        Body: {current_template.get('body', '')}
        Signature: {current_template.get('signature', '')}
        
        ## Contact Information (extracted from job description):
        Email: {contact_info['email']}
        Name: {contact_info['name']}
        Company: {contact_info['company']}
        
        Please create:
        1. A brief, conversational email that sounds like a real person wrote it
        2. Highlight 2-3 most relevant skills/experiences matching the job requirements
        3. Keep it concise and enthusiastic
        4. Maintain the greeting/body/signature structure
        5. Keep the placeholders {{name}} and {{position}} where appropriate
        6. Mention the company name ({contact_info['company']}) in the body if available
        
        Also suggest values for:
        - Position/designation (extracted from job description)
        - Employer/company name (use extracted company name if available: {contact_info['company']})
        - Subject line
        - Recipient email (use the extracted email if available)
        
        Return as JSON with fields: greeting, body, signature, position, employer, subject, recipient_email, company
        """

        st.info(f"Sending request to AI model: {model}...")

        # Check if request was cancelled
        if cancel_request:
            request_in_progress = False
            return {"success": False, "error": "Request cancelled by user"}

        # Call the API
        response = client.chat.completions.create(
            model=model,  # Use the selected model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        # Extract the content
        result = response.choices[0].message.content

        # Check if request was cancelled
        if cancel_request:
            request_in_progress = False
            return {"success": False, "error": "Request cancelled by user"}

        st.info("Received response from AI model, processing...")

        # Try to parse as JSON
        try:
            # Find JSON in the response (in case there's additional text)
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                template_data = json.loads(json_str)
            else:
                # Fallback: try to parse the whole response
                template_data = json.loads(result)

            # Check if request was cancelled
            if cancel_request:
                request_in_progress = False
                return {"success": False, "error": "Request cancelled by user"}

            # Set request completed
            request_in_progress = False

            return {
                "success": True,
                "template": {
                    "greeting": template_data.get(
                        "greeting", current_template.get("greeting", "")
                    ),
                    "body": template_data.get("body", current_template.get("body", "")),
                    "signature": template_data.get(
                        "signature", current_template.get("signature", "")
                    ),
                    "position": template_data.get("position", ""),
                    "employer": template_data.get("employer", contact_info["company"]),
                    "company": template_data.get("company", contact_info["company"]),
                    "subject": template_data.get("subject", ""),
                    "recipient_email": template_data.get(
                        "recipient_email", contact_info["email"]
                    ),
                },
            }
        except json.JSONDecodeError as e:
            st.warning(f"JSON parsing failed: {str(e)}. Trying manual extraction...")
            st.code(result)

            # If JSON parsing fails, extract parts manually
            lines = result.split("\n")
            greeting = ""
            body = ""
            signature = ""
            position = ""
            employer = contact_info["company"] or ""
            company = contact_info["company"] or ""
            subject = ""
            recipient_email = contact_info["email"]
            current_section = None

            for line in lines:
                line = line.strip()
                if line.lower().startswith("greeting:"):
                    current_section = "greeting"
                    greeting = line[len("greeting:") :].strip()
                elif line.lower().startswith("body:"):
                    current_section = "body"
                    body = line[len("body:") :].strip()
                elif line.lower().startswith("signature:"):
                    current_section = "signature"
                    signature = line[len("signature:") :].strip()
                elif line.lower().startswith("position:"):
                    position = line[len("position:") :].strip()
                elif line.lower().startswith("employer:"):
                    employer = line[len("employer:") :].strip()
                elif line.lower().startswith("company:"):
                    company = line[len("company:") :].strip()
                elif line.lower().startswith("subject:"):
                    subject = line[len("subject:") :].strip()
                elif line.lower().startswith("recipient_email:"):
                    recipient_email = line[len("recipient_email:") :].strip()
                elif current_section == "greeting" and not line.lower().startswith(
                    "body:"
                ):
                    greeting += "\n" + line
                elif current_section == "body" and not line.lower().startswith(
                    "signature:"
                ):
                    body += "\n" + line
                elif current_section == "signature":
                    signature += "\n" + line

            # Check if request was cancelled
            if cancel_request:
                request_in_progress = False
                return {"success": False, "error": "Request cancelled by user"}

            # Set request completed
            request_in_progress = False

            return {
                "success": True,
                "template": {
                    "greeting": greeting or current_template.get("greeting", ""),
                    "body": body or current_template.get("body", ""),
                    "signature": signature or current_template.get("signature", ""),
                    "position": position,
                    "employer": employer,
                    "company": company,
                    "subject": subject,
                    "recipient_email": recipient_email,
                },
            }

    except Exception as e:
        st.error(f"Error in generate_improved_template: {str(e)}")
        st.error(traceback.format_exc())
        # Set request completed
        request_in_progress = False
        return {"success": False, "error": str(e)}


def cancel_ai_request():
    """Cancel an in-progress AI request"""
    global cancel_request, request_in_progress

    if request_in_progress:
        cancel_request = True
        st.warning("Cancelling AI request...")
        return True
    return False


def is_request_in_progress():
    """Check if a request is in progress"""
    global request_in_progress
    return request_in_progress
