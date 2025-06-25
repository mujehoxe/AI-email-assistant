# Email Generator

An AI-powered application to help create personalized job application emails based on job descriptions and your resume.

## Features

- Upload your resume (PDF, DOCX) and automatically extract text
- Generate personalized emails based on job descriptions
- Persist resume data between sessions
- Extract company name, contact name, and email from job descriptions
- Save and manage email templates
- Select from multiple AI models (requires G4F API server)
- Directly send emails from the application
- Copy generated email to clipboard

## Setup Instructions

### Local Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your email credentials (see `.env.example`)
4. Run the application:
   ```
   streamlit run app.py
   ```

### Using Docker

1. Build the Docker image:
   ```
   docker build -t email-generator .
   ```
2. Run the container:
   ```
   docker run -p 8501:8501 -p 1337:1337 -v $(pwd)/data:/app/data email-generator
   ```

### Deployment on render.com

1. Fork or clone this repository to your GitHub account
2. Sign up for a [render.com](https://render.com) account
3. Create a new Web Service and connect your repository
4. Select "Docker" as the environment
5. Under Advanced, add the following environment variables:
   - `EMAIL_ADDRESS`: Your email address
   - `EMAIL_PASSWORD`: Your email app password
   - `SMTP_SERVER`: Your SMTP server (e.g., smtp.gmail.com)
   - `SMTP_PORT`: Your SMTP port (usually 587)
6. Click "Create Web Service"

**Note:** On render.com, the G4F API server runs inside the same container as the Streamlit app, so you don't need to configure separate ports.

## Docker Setup for G4F API

If you want to run only the G4F API server (advanced setup):

```bash
docker run \
  -p 1337:8080 -p 8080:8080 \
  -v ${PWD}/har_and_cookies:/app/har_and_cookies \
  -v ${PWD}/generated_media:/app/generated_media \
  hlohaus789/g4f:latest-slim
```

Then configure `G4F_API_HOST` and `G4F_API_PORT` in your `.env` file to point to this server.

## Troubleshooting

### Email Sending Issues

- For Gmail, create an app password: [Google Account Help](https://support.google.com/accounts/answer/185833)
- Check if your SMTP settings are correct

### G4F API Connection

- If the API status shows "Not Running", try using the "Start G4F API Server" button
- If models aren't loading, click "Refresh Models List"
- Check if port 1337 is accessible and not blocked by your network or firewall

### Resume Parsing

- Make sure PyPDF2 and python-docx are installed correctly
- Only PDF and DOCX formats are supported
