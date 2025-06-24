import streamlit as st
from openai import OpenAI
import json
import subprocess
import time
import threading
import os
import traceback
import re

# Global flag to check if API server is running
api_server_running = False
# Global flag to check if AI request should be canceled
request_canceled = False


def start_g4f_api_server():
    """Start the G4F API server in a separate thread"""
    global api_server_running

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


def get_ai_client():
    """Get OpenAI client configured to use G4F API"""
    # Initialize the OpenAI client
    client = OpenAI(
        api_key="secret",  # Using "secret" as the API key since G4F doesn't require a real one
        base_url="http://localhost:1337/v1",  # Point to the G4F API endpoint
    )
    return client


def extract_email_from_job_description(job_description):
    """Extract email addresses from job description"""
    # Regular expression for email extraction
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    emails = re.findall(email_pattern, job_description)
    return emails[0] if emails else ""


def reset_cancel_flag():
    """Reset the cancel flag"""
    global request_canceled
    request_canceled = False


def cancel_ai_request():
    """Cancel the current AI request"""
    global request_canceled
    request_canceled = True
    st.warning("AI request canceled by user")
    return True


def generate_improved_template(job_description, resume_content, current_template):
    """Generate an improved template based on job description and resume"""
    # We only need to read the global variable, not modify it
    reset_cancel_flag()

    try:
        # Extract email from job description
        recipient_email = extract_email_from_job_description(job_description)

        # Ensure API server is running
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
        
        Please create:
        1. A brief, conversational email that sounds like a real person wrote it
        2. Highlight 2-3 most relevant skills/experiences matching the job requirements
        3. Keep it concise and enthusiastic
        4. Maintain the greeting/body/signature structure
        5. Keep the placeholders {{name}} and {{position}} where appropriate
        
        Also extract or suggest values for:
        - Recipient email (from job description): {recipient_email if recipient_email else "Not found, please suggest"}
        - Position/designation (extracted from job description)
        - Employer name (company from job description)
        - Subject line
        
        Return as JSON with fields: greeting, body, signature, position, employer, subject, recipient_email
        """

        st.info("Sending request to AI model...")

        # Check if request was canceled
        if request_canceled:
            return {"success": False, "error": "Request canceled by user"}

        # Call the API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using gpt-4o-mini for better quality
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        # Check if request was canceled
        if request_canceled:
            return {"success": False, "error": "Request canceled by user"}

        # Extract the content
        result = response.choices[0].message.content

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
                    "employer": template_data.get("employer", ""),
                    "subject": template_data.get("subject", ""),
                    "recipient_email": template_data.get(
                        "recipient_email", recipient_email
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
            employer = ""
            subject = ""
            extracted_email = recipient_email
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
                elif line.lower().startswith("subject:"):
                    subject = line[len("subject:") :].strip()
                elif line.lower().startswith(
                    "recipient_email:"
                ) or line.lower().startswith("recipient email:"):
                    email_part = line.split(":", 1)[1].strip()
                    if "@" in email_part:  # Simple validation
                        extracted_email = email_part
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

            return {
                "success": True,
                "template": {
                    "greeting": greeting or current_template.get("greeting", ""),
                    "body": body or current_template.get("body", ""),
                    "signature": signature or current_template.get("signature", ""),
                    "position": position,
                    "employer": employer,
                    "subject": subject,
                    "recipient_email": extracted_email,
                },
            }

    except Exception as e:
        st.error(f"Error in generate_improved_template: {str(e)}")
        st.error(traceback.format_exc())
        return {"success": False, "error": str(e)}
