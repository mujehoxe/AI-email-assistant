import streamlit as st
import json
import subprocess
import time
import threading
import os
import traceback
import re
import requests
import openai
from typing import Dict, List, Any, Optional, Union, Tuple

# Configuration
G4F_API_HOST = os.environ.get("G4F_API_HOST", "localhost")
G4F_API_PORT = int(os.environ.get("G4F_API_PORT", "8080"))
G4F_API_BASE_URL = f"http://{G4F_API_HOST}:{G4F_API_PORT}/v1"

# Default model and fallback models
DEFAULT_MODEL = "llama-4-scout"
FALLBACK_MODELS = [
    "llama-4-scout",  # Default
    "gpt-4o",
    "gpt-3.5-turbo",
    "gpt-4",
    "claude-3-opus",
    "claude-3-sonnet",
    "gemini-pro",
    "mistral-medium",
]

# Global state
api_server_running = False
request_in_progress = False
cancel_request = False
AVAILABLE_MODELS = FALLBACK_MODELS.copy()

# Type definitions
ContactInfo = Dict[str, str]
TemplateData = Dict[str, str]
APIResponse = Dict[str, Any]


def extract_email_from_text(text: str) -> str:
    """Extract email addresses from text"""
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else ""


def extract_company_name(job_description: str) -> str:
    """Extract company name from job description"""
    patterns = [
        r"(?:at|with|for|join)\s+([\w\s&\-\.]+?)(?:is\s+looking|is\s+seeking|is\s+hiring|is\s+searching|has\s+an\s+opening|has\s+a\s+job|has\s+a\s+position|has\s+an\s+opportunity)",
        r"(?:at|with|for|join)\s+([\w\s&\-\.]+?)(?:\.|,|\sin\s)",
        r"([\w\s&\-\.]+?)(?:\sis\s+looking|\sis\s+seeking|\sis\s+hiring|\shas\s+an\s+opening|\shas\s+a\s+job|\shas\s+a\s+position)",
        r"(?:company|employer):\s*([\w\s&\-\.]+)",
        r"(?:about\s+us|about\s+the\s+company|company\s+overview|about\s+the\s+team)\s*(?:\n|\r\n?)([\w\s&\-\.]+)",
        r"(?:about\s+)([\w\s&\-\.]+)(?:\s+[\w\s&\-\.]+\s+is\s+a)",
    ]

    for pattern in patterns:
        matches = re.search(pattern, job_description, re.IGNORECASE)
        if matches:
            company_name = matches.group(1).strip()
            company_name = re.sub(r"\s+", " ", company_name)
            if company_name:
                return company_name

    # If no company name found, try to extract from emails in the job description
    emails = extract_email_from_text(job_description)
    if emails:
        # Try to get company name from email domain
        email = emails if isinstance(emails, str) else emails[0]
        domain = email.split('@')[-1]
        if domain:
            # Remove TLD (.com, .org, etc.) and convert to title case
            company = domain.split('.')[0].title()
            if company and len(company) > 2:  # Ensure it's a meaningful name
                return company

    return ""


def extract_contact_info(job_description: str) -> ContactInfo:
    """Extract contact information from job description

    This function only extracts information from the job description, not the resume.
    It returns a dictionary with email, name, and company extracted from the job description.
    """
    email = extract_email_from_text(job_description)
    company_name = extract_company_name(job_description)
    name = ""

    # Extract name based on email if available
    name_patterns = []
    if email:
        name_patterns.extend(
            [
                r"(?:contact|send|email|apply|resume to|cv to|application to)[:\s]+([\w\s]+)[\s\(<]+"
                + re.escape(email),
                r"([\w\s]+)[\s\(<]+" + re.escape(email),
            ]
        )

    # General name patterns
    name_patterns.extend(
        [
            r"(?:contact|send|email|apply|resume to|cv to|application to)[:\s]+([\w\s]+)",
            r"(?:recruiter|hiring manager|hr manager|contact person)[:\s]+([\w\s]+)",
        ]
    )

    # Try to find name in job description
    for pattern in name_patterns:
        matches = re.search(pattern, job_description, re.IGNORECASE)
        if matches:
            name = matches.group(1).strip()
            break

    # If name not found in job description but we have email, extract from email
    if not name and email:
        # Try to extract name from the email address
        local_part = email.split('@')[0]
        
        # Try different common email formats
        if '.' in local_part:  # firstname.lastname@domain.com
            parts = local_part.split('.')
            name = ' '.join(part.title() for part in parts)
        elif '_' in local_part:  # firstname_lastname@domain.com
            parts = local_part.split('_')
            name = ' '.join(part.title() for part in parts)
        elif '-' in local_part:  # firstname-lastname@domain.com
            parts = local_part.split('-')
            name = ' '.join(part.title() for part in parts)
        else:
            # Try to detect if it's a name by checking if it contains digits
            if not any(c.isdigit() for c in local_part):
                name = local_part.title()
        
        # If we couldn't extract a name from email, fall back to utils.extract_name
        if not name:
            from utils import extract_name
            name = extract_name(email)

    # Filter out generic or invalid names
    generic_names = [
        "at",
        "hr",
        "info",
        "jobs",
        "careers",
        "admin",
        "recruitment",
        "recruiting",
        "contact",
        "apply",
        "job",
        "resume",
        "application",
        "hiring",
        "talent",
        "human resources",
        "humanresources",
    ]

    # Check if the extracted name is generic
    if name.lower() in generic_names or len(name) <= 2:
        name = ""  # Clear the name if it's generic

    # If company name is not found but we have email, try to extract from email domain
    if not company_name and email:
        domain = email.split('@')[-1]
        if domain:
            # Remove TLD (.com, .org, etc.) and convert to title case
            company = domain.split('.')[0].title()
            if company and len(company) > 2:  # Ensure it's a meaningful name
                company_name = company

    return {"email": email, "name": name, "company": company_name}


def check_api_server_status() -> bool:
    """Check if the G4F API server is running"""
    try:
        response = requests.get(
            f"http://{G4F_API_HOST}:{G4F_API_PORT}/v1/models", timeout=5
        )
        return response.status_code == 200
    except (requests.RequestException, ConnectionError):
        return False


def get_available_models() -> List[str]:
    """Get list of available models from G4F API"""
    if not check_api_server_status():
        return FALLBACK_MODELS.copy()

    try:
        response = requests.get(
            f"http://{G4F_API_HOST}:{G4F_API_PORT}/v1/models", timeout=5
        )

        if response.status_code == 200:
            data = response.json()

            if "data" in data and isinstance(data["data"], list):
                models = []
                for model in data["data"]:
                    if isinstance(model, dict) and "id" in model:
                        model_id = model["id"]
                        if model_id != "default" and model_id not in models:
                            models.append(model_id)

                if models:
                    return models

        return FALLBACK_MODELS.copy()

    except Exception as e:
        st.warning(f"Could not fetch available models from G4F API: {str(e)}")
        return FALLBACK_MODELS.copy()


def start_g4f_api_server() -> None:
    """Start the G4F API server in a separate thread"""
    global api_server_running, AVAILABLE_MODELS

    if api_server_running:
        return

    def run_server():
        try:
            st.info("Starting G4F API server...")
            process = subprocess.Popen(
                ["python", "-c", "from g4f.api import run_api; run_api()"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            for line in process.stderr:
                st.error(f"G4F API server error: {line.decode().strip()}")
            for line in process.stdout:
                st.info(f"G4F API server: {line.decode().strip()}")

        except Exception as e:
            st.error(f"Failed to start G4F API server: {str(e)}")
            st.error(traceback.format_exc())

    # Start server in a separate thread
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()

    # Wait for server to start
    time.sleep(5)
    api_server_running = True

    # Update session state
    st.session_state.api_server_status = "Running"

    # Update available models
    time.sleep(2)  # Wait for server initialization
    try:
        fresh_models = get_available_models()
        if fresh_models:
            AVAILABLE_MODELS = fresh_models
            if "ai_models" in st.session_state:
                st.session_state.ai_models = fresh_models
    except Exception as e:
        st.warning(f"Could not update model list: {e}")


def get_ai_client():
    """Get OpenAI-compatible client configured to use G4F API"""
    base_url = f"http://{G4F_API_HOST}:{G4F_API_PORT}/v1"

    try:
        client = openai.OpenAI(
            api_key="not-needed",
            base_url=base_url,
        )
        return client
    except Exception as e:
        try:
            import g4f

            return g4f
        except ImportError:
            st.error("Failed to import g4f. Make sure g4f is installed.")
            return None


def get_g4f_model_from_name(model_name: str):
    """Map OpenAI model name to g4f provider if possible"""
    try:
        import g4f
        from g4f.models import (
            gpt_4,
            gpt_4_turbo,
            claude_3_opus,
            claude_3_sonnet,
            gemini_pro,
        )
        
        model_map = {
            "gpt-4o": gpt_4_turbo,  # Using gpt_4_turbo as fallback for gpt-4o
            "gpt-3.5-turbo": gpt_4,  # Using gpt_4 as fallback since gpt_35_turbo doesn't exist
            "gpt-4": gpt_4,
            "gpt-4-turbo": gpt_4_turbo,
            "claude-3-opus": claude_3_opus,
            "claude-3-sonnet": claude_3_sonnet,
            "gemini-pro": gemini_pro,
        }

        return model_map.get(model_name, gpt_4)  # Default to gpt_4 if model not found
    except ImportError:
        st.error("Failed to import g4f. Make sure g4f is installed.")
        return None


def create_prompt(
    job_description: str,
    resume_content: str,
    current_template: Dict[str, str],
    contact_info: ContactInfo,
) -> str:
    """Create prompt for AI model"""
    # Extract first name if full name is available
    first_name = ""
    last_name = ""
    if contact_info["name"]:
        name_parts = contact_info["name"].split()
        if name_parts:
            first_name = name_parts[0]
            if len(name_parts) > 1:
                last_name = name_parts[-1]

    return f"""
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
    First Name: {first_name}
    Last Name: {last_name}
    Company: {contact_info['company'] or 'Not found (which is okay)'}
    
    Please create:
    1. A brief, conversational email that sounds like a real person wrote it
    2. Highlight 2-3 most relevant skills/experiences matching the job requirements
    3. Keep it concise and enthusiastic
    4. Maintain the greeting/body/signature structure
    5. In the greeting:
       - If a valid recipient name is found, use their first name (preferred) or last name with appropriate title (Mr./Ms.)
       - If no valid name is found (or name is generic like "at", "hr", etc.), use "Hiring Manager" in the greeting
       - DO NOT use a full name in the greeting
    6. Keep the placeholder {{{{name}}}} in the greeting, but make sure the instructions specify to use first or last name only
    7. Keep the placeholder {{{{position}}}} where appropriate
    8. If a company name was detected, mention it in the body. If no company name was found, try to extract it from the email domain (e.g., "acme" from "jobs@acme.com")
    
    Important: 
    - Only extract information from the job description, not the resume
    - Use the resume only to match skills and experience to the job requirements
    - If you don't have a company name, try to extract it from the email domain if available
    - If you can't determine a company name, leave the company field empty. DO NOT use placeholder text like "the company" or "[Company Name]"
    - If the recipient name is missing, try to extract it from the email address (e.g., "John Smith" from "john.smith@company.com")
    - If the recipient name is still missing, invalid, or generic (like "at", "hr", "info", "jobs", "careers", "admin", etc.), leave the recipient_name field empty in your response and use "Hiring Manager" in the greeting
    
    Format your response as a valid JSON object with the following structure:
    
    {{
      "greeting": "Your greeting here",
      "body": "Your email body here",
      "signature": "Your signature here",
      "position": "Extracted position from job description",
      "employer": "Extracted employer name from job description or email domain if found",
      "subject": "Suggested email subject",
      "recipient_name": "Extracted recipient name from job description or email if found (leave empty if invalid or generic)",
      "recipient_email": "Extracted email from job description if found"
    }}
    """


def parse_ai_response(
    result: str, contact_info: ContactInfo
) -> Tuple[bool, Dict[str, Any]]:
    """Parse the AI response and extract template data

    Uses contact_info from the job description as fallback if the AI doesn't provide values.
    """
    # Try to parse the JSON directly
    try:
        template = json.loads(result)

        # Check if template has all required fields
        required_fields = ["greeting", "body", "signature"]
        if all(field in template for field in required_fields):
            return True, {"success": True, "template": template}
        else:
            missing = [field for field in required_fields if field not in template]
            return False, {
                "success": False,
                "error": f"Response missing required fields: {', '.join(missing)}",
                "template": template,
            }

    except json.JSONDecodeError:
        # Try to extract JSON from markdown
        try:
            json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            match = re.search(json_pattern, result)

            if match:
                json_str = match.group(1)
                template = json.loads(json_str)

                required_fields = ["greeting", "body", "signature"]
                if all(field in template for field in required_fields):
                    return True, {"success": True, "template": template}
                else:
                    missing = [
                        field for field in required_fields if field not in template
                    ]
                    return False, {
                        "success": False,
                        "error": f"Response missing required fields: {', '.join(missing)}",
                        "template": template,
                    }
        except:
            pass

        # Manual extraction as last resort
        st.warning("JSON parsing failed. Trying manual extraction...")

        lines = result.split("\n")
        template = {
            "greeting": "",
            "body": "",
            "signature": "",
            "position": "",
            "employer": "",
            "subject": "",
            "recipient_email": contact_info["email"],
            "recipient_name": contact_info["name"],
        }

        current_section = None

        for line in lines:
            line = line.strip()
            if line.lower().startswith("greeting:"):
                current_section = "greeting"
                template["greeting"] = line[len("greeting:") :].strip()
            elif line.lower().startswith("body:"):
                current_section = "body"
                template["body"] = line[len("body:") :].strip()
            elif line.lower().startswith("signature:"):
                current_section = "signature"
                template["signature"] = line[len("signature:") :].strip()
            elif line.lower().startswith("position:"):
                template["position"] = line[len("position:") :].strip()
            elif line.lower().startswith("employer:"):
                template["employer"] = line[len("employer:") :].strip()
            elif line.lower().startswith("company:"):
                if not template["employer"]:
                    template["employer"] = line[len("company:") :].strip()
            elif line.lower().startswith("subject:"):
                template["subject"] = line[len("subject:") :].strip()
            elif line.lower().startswith("recipient_email:"):
                template["recipient_email"] = line[len("recipient_email:") :].strip()
            elif line.lower().startswith("recipient_name:"):
                template["recipient_name"] = line[len("recipient_name:") :].strip()
            elif current_section == "greeting":
                template["greeting"] += "\n" + line
            elif current_section == "body":
                template["body"] += "\n" + line
            elif current_section == "signature":
                template["signature"] += "\n" + line

        # Check if we have the required fields
        if template["greeting"] and template["body"] and template["signature"]:
            return True, {"success": True, "template": template}
        else:
            missing = []
            if not template["greeting"]:
                missing.append("greeting")
            if not template["body"]:
                missing.append("body")
            if not template["signature"]:
                missing.append("signature")

            return False, {
                "success": False,
                "error": f"Failed to extract required fields: {', '.join(missing)}",
                "template": template,
            }


def generate_improved_template(
    job_description: str,
    resume_content: str,
    current_template: Dict[str, str],
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Generate an improved template based on job description and resume"""
    global request_in_progress, cancel_request

    try:
        # Set request in progress flag
        request_in_progress = True
        cancel_request = False

        # Extract contact info from job description
        contact_info = extract_contact_info(job_description)

        # Check API server status
        api_server_running = check_api_server_status()

        # Create prompt
        prompt = create_prompt(
            job_description, resume_content, current_template, contact_info
        )

        # Get client for API request
        client = get_ai_client()

        # Generate response
        result = ""
        if api_server_running and hasattr(client, "chat"):
            # Using OpenAI-compatible API
            st.info(f"Sending request to AI model: {model}...")

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                )

                result = response.choices[0].message.content
            except Exception as api_error:
                st.error(f"API error: {str(api_error)}")
                # Fall back to direct g4f
                api_server_running = False

        if not api_server_running:
            # Using g4f directly as fallback
            st.warning("Using g4f directly (this may take longer)...")

            try:
                import g4f

                g4f_model = get_g4f_model_from_name(model)
                if not g4f_model:
                    return {"success": False, "error": "Failed to get g4f model"}

                result = g4f.ChatCompletion.create(
                    model=g4f_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                )
            except Exception as g4f_error:
                st.error(f"Error using g4f directly: {str(g4f_error)}")
                return {"success": False, "error": str(g4f_error)}

        # Check if request was cancelled
        if cancel_request:
            return {"success": False, "error": "Request cancelled by user"}

        # Parse the response
        success, response_data = parse_ai_response(result, contact_info)
        return response_data

    except Exception as e:
        st.error(f"Error in generate_improved_template: {str(e)}")
        st.error(traceback.format_exc())
        return {"success": False, "error": str(e)}
    finally:
        # Reset request in progress flag
        request_in_progress = False


def cancel_ai_request() -> bool:
    """Cancel an in-progress AI request"""
    global cancel_request, request_in_progress

    if request_in_progress:
        cancel_request = True
        return True
    return False


def is_request_in_progress() -> bool:
    """Check if a request is in progress"""
    global request_in_progress
    return request_in_progress
