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
import openai

# Get G4F API endpoint from environment variables or use default
G4F_API_HOST = os.environ.get("G4F_API_HOST", "localhost")
G4F_API_PORT = int(os.environ.get("G4F_API_PORT", "1337"))
G4F_API_BASE_URL = f"http://{G4F_API_HOST}:{G4F_API_PORT}/v1"

# Global flag to check if API server is running
api_server_running = False
# Global flag to check if a request is in progress
request_in_progress = False
# Global flag to cancel a request
cancel_request = False

# Default model to use
DEFAULT_MODEL = "gpt-4o"

# Fallback models in case we can't fetch from API
FALLBACK_MODELS = [
    "gpt-4o",  # Keep this first as the default
    "gpt-4o-mini",  # Keep this first as the default
    "gpt-3.5-turbo",
    "gpt-4",
    "claude-3-opus",
    "claude-3-sonnet",
    "gemini-pro",
    "mistral-medium",
]


def get_available_models():
    """Get list of available models from G4F API"""
    global FALLBACK_MODELS, G4F_API_HOST, G4F_API_PORT

    # Check if API server is running
    if not check_api_server_status():
        st.warning("G4F API server is not running. Using fallback models.")
        return FALLBACK_MODELS

    try:
        # Fetch models from G4F API
        response = requests.get(f"http://{G4F_API_HOST}:{G4F_API_PORT}/v1/models")

        if response.status_code == 200:
            data = response.json()

            # Extract model names based on the API response format
            if "data" in data and isinstance(data["data"], list):
                # Extract unique model IDs
                models = []
                for model in data["data"]:
                    if isinstance(model, dict) and "id" in model:
                        model_id = model["id"]
                        # Filter out 'default' model and avoid duplicates
                        if model_id != "default" and model_id not in models:
                            models.append(model_id)

                # Return models if found, otherwise fallback
                if models:
                    return models

        # If we reached here, something went wrong with parsing
        st.warning(
            "Could not parse models from G4F API response. Using fallback models."
        )
        return FALLBACK_MODELS

    except Exception as e:
        st.warning(f"Could not fetch available models from G4F API: {str(e)}")
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
    # Look for common patterns that indicate company name
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
            # Get the company name from the matched group
            company_name = matches.group(1).strip()
            # Clean up company name
            company_name = re.sub(r"\s+", " ", company_name)
            if company_name:
                return company_name

    # If no company name found, return empty string
    return ""


def extract_contact_info(job_description):
    """Extract contact information from job description"""
    # Extract email
    email = extract_email_from_text(job_description)

    # Extract company name
    company_name = extract_company_name(job_description)

    # Try to extract name associated with the email
    name = ""

    # Look for common patterns like "Contact: John Doe (john.doe@example.com)"
    name_patterns = []

    # Only add email-based patterns if we have an email
    if email:
        name_patterns.extend(
            [
                r"(?:contact|send|email|apply|resume to|cv to|application to)[:\s]+([\w\s]+)[\s\(<]+"
                + re.escape(email),
                r"([\w\s]+)[\s\(<]+" + re.escape(email),
            ]
        )

    # Add general name patterns
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
        # Extract name from email address
        from utils import extract_name

        name = extract_name(email)

    # Return extracted information - no default values
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
    global G4F_API_HOST, G4F_API_PORT

    try:
        response = requests.get(f"http://{G4F_API_HOST}:{G4F_API_PORT}/v1/models")
        return response.status_code == 200
    except:
        return False


def get_ai_client():
    """Get OpenAI-compatible client configured to use G4F API"""
    global G4F_API_HOST, G4F_API_PORT

    # Use environment variable or default to localhost:1337
    base_url = f"http://{G4F_API_HOST}:{G4F_API_PORT}/v1"

    try:
        # Create an OpenAI client with the G4F API base URL
        client = openai.OpenAI(
            api_key="not-needed",  # G4F doesn't require a real API key
            base_url=base_url,
        )
        return client
    except Exception as e:
        st.error(f"Error creating OpenAI client: {str(e)}")
        # Fallback to using g4f directly
        try:
            import g4f

            st.info("Using g4f directly as fallback")
            return g4f
        except ImportError:
            st.error("Failed to import g4f. Make sure g4f is installed.")
            return None


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

        # Extract first name if full name is available
        first_name = ""
        if contact_info["name"]:
            name_parts = contact_info["name"].split()
            if name_parts:
                first_name = name_parts[0]

        # Add company name to log if found
        if contact_info["company"]:
            st.info(f"Detected company name: {contact_info['company']}")
        else:
            st.info("No company name detected in the job description, which is okay")

        # Ensure API server is running
        api_server_running = check_api_server_status()

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
        First Name: {first_name}
        Company: {contact_info['company'] or 'Not found (which is okay)'}
        
        Please create:
        1. A brief, conversational email that sounds like a real person wrote it
        2. Highlight 2-3 most relevant skills/experiences matching the job requirements
        3. Keep it concise and enthusiastic
        4. Maintain the greeting/body/signature structure
        5. In the greeting, use either first name (preferred) or last name with appropriate title (Mr./Ms.) - NOT full name
        6. Keep the placeholder {{{{name}}}} in the greeting, but make sure the instructions specify to use first or last name only
        7. Keep the placeholder {{{{position}}}} where appropriate
        8. If a company name was detected, mention it in the body. If no company name was found, that's okay - don't use a placeholder
        
        Important: If you don't have a company name, leave the company field empty. Do NOT use placeholder text like "the company" or "[Company Name]".
        
        Also suggest values for:
        - Position/designation (extracted from job description)
        - Employer/company name (use extracted company name ONLY if available: {contact_info['company'] or ''})
        
        Format your response as a valid JSON object with the following structure:
        
        {{{{
          "greeting": "Your greeting here",
          "body": "Your email body here",
          "signature": "Your signature here",
          "position": "Extracted position",
          "employer": "Extracted employer name if found",
          "subject": "Suggested email subject"
        }}}}
        """

        # Get client for API request
        client = get_ai_client()

        if api_server_running and hasattr(client, "chat"):
            # Using OpenAI-compatible API
            st.info(f"Sending request to AI model: {model}...")

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )

            # Extract response
            result = response.choices[0].message.content
        else:
            # Using g4f directly as fallback
            st.warning(
                "API server not available, using g4f directly (this may take longer)..."
            )

            try:
                import g4f

                # Map the model name to a g4f provider if possible
                g4f_model = get_g4f_model_from_name(model)

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

        # Parse the response
        try:
            # Try to parse the JSON directly
            template = json.loads(result)

            # Check if template has all required fields
            required_fields = ["greeting", "body", "signature"]
            if all(field in template for field in required_fields):
                return {"success": True, "template": template}
            else:
                missing = [field for field in required_fields if field not in template]
                return {
                    "success": False,
                    "error": f"Response missing required fields: {', '.join(missing)}",
                    "template": template,
                }

        except json.JSONDecodeError:
            # If direct JSON parsing fails, try to extract JSON from markdown
            try:
                # Look for JSON block in markdown
                json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
                match = re.search(json_pattern, result)

                if match:
                    json_str = match.group(1)
                    template = json.loads(json_str)

                    # Check if template has all required fields
                    required_fields = ["greeting", "body", "signature"]
                    if all(field in template for field in required_fields):
                        return {"success": True, "template": template}
                    else:
                        missing = [
                            field for field in required_fields if field not in template
                        ]
                        return {
                            "success": False,
                            "error": f"Response missing required fields: {', '.join(missing)}",
                            "template": template,
                        }
            except:
                # If still can't parse JSON, do manual extraction
                pass

            # If JSON parsing fails, extract parts manually
            st.warning("JSON parsing failed. Trying manual extraction...")

            lines = result.split("\n")
            greeting = ""
            body = ""
            signature = ""
            position = ""
            employer = ""
            company = ""
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
                elif current_section == "greeting":
                    greeting += "\n" + line
                elif current_section == "body":
                    body += "\n" + line
                elif current_section == "signature":
                    signature += "\n" + line

            template = {
                "greeting": greeting,
                "body": body,
                "signature": signature,
                "position": position,
                "employer": employer or company,
                "subject": subject,
                "recipient_email": recipient_email,
            }

            return {"success": True, "template": template}

    except Exception as e:
        st.error(f"Error in generate_improved_template: {str(e)}")
        st.error(traceback.format_exc())
        return {"success": False, "error": str(e)}
    finally:
        # Reset request in progress flag
        request_in_progress = False


def get_g4f_model_from_name(model_name):
    """Map OpenAI model name to g4f provider if possible"""
    import g4f

    # Map common model names to g4f providers
    model_map = {
        "gpt-4o": g4f.models.gpt_4o,
        "gpt-4o-mini": g4f.models.gpt_4o_mini,
        "gpt-3.5-turbo": g4f.models.gpt_35_turbo,
        "gpt-4": g4f.models.gpt_4,
        "gpt-4-turbo": g4f.models.gpt_4_turbo,
        "claude-3-opus": g4f.models.claude_3_opus,
        "claude-3-sonnet": g4f.models.claude_3_sonnet,
        "gemini-pro": g4f.models.gemini_pro,
    }

    # Return mapped model or default to gpt-3.5-turbo
    return model_map.get(model_name, g4f.models.gpt_35_turbo)


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
