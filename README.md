# Email Generator

An AI-powered application to help create personalized job application emails based on job descriptions and your resume.

## Features

- **Resume Processing**: Upload your resume (PDF, DOCX) and automatically extract text
- **AI Resume Customization**: Generate tailored resumes that match specific job descriptions
- **Personalized Email Generation**: Create custom job application emails based on job descriptions and your resume
- **Smart Information Extraction**: Automatically extract company name, contact name, and email from job descriptions
- **Resume & Email Management**: Persist resume data between sessions and save email templates
- **Multiple AI Models**: Select from various AI models (requires G4F API server)
- **Direct Email Sending**: Send emails directly from the application with resume attachments
- **Easy Export**: Download customized resumes and copy generated emails to clipboard

## How to Use

### Multi-Page Interface Workflow

The application features a clean, organized 5-tab interface that guides you through the complete process:

#### **📄 Tab 1: Resume Upload**
1. **Upload Resume**: Upload your resume in PDF or DOCX format
2. **Preview Content**: Review extracted text from your resume
3. **Manage Files**: Clear or replace your resume as needed

#### **📝 Tab 2: Job Description**
1. **Enter Job Description**: Paste the complete job description you're applying for
2. **Quick Analysis**: View automatically detected company, contact info, and word count
3. **Validation**: System ensures job description is complete before proceeding

#### **🎯 Tab 3: Resume Customization**
1. **AI Customization**: Generate a tailored resume that matches the job description
   - Reorganizes and emphasizes relevant experience
   - Adds professional summary and relevant job title
   - Incorporates keywords from the job description
   - **Smart Format Preservation**: Maintains Word document format if original is DOCX
2. **Choose Version**: Select between original or customized resume
3. **Download Options**: Export as text or professionally formatted Word document

#### **📧 Tab 4: Template Generation**
1. **AI Model Selection**: Choose from available AI models for template generation
2. **Resume Selection**: Confirm which resume version to use (original or customized)
3. **Generate Template**: Create personalized email template with extracted information
4. **Template Preview**: Review the generated greeting, body, and signature

#### **✉️ Tab 5: Email Sending**
1. **Configure Recipients**: Add email, name, company, and position details
2. **Customize Content**: Edit the email subject, greeting, body, and signature
3. **Smart Attachments**: Automatically attaches the appropriate resume format
4. **Send or Export**: Send email directly or copy content to clipboard
5. **Save Template**: Save successful templates for future use

### Key Benefits of Resume Customization

- **Better ATS Matching**: Incorporates relevant keywords from job descriptions
- **Targeted Positioning**: Highlights your most relevant experience for each role
- **Professional Summary**: Adds a compelling summary tailored to the specific position
- **Smart Format Preservation**: Maintains Word document format when original is DOCX
- **Multiple Export Options**: Download as text or Word document for flexibility
- **Truthful Enhancement**: Only reorganizes and emphasizes existing information - never fabricates experience
- **Streamlined Workflow**: Organized multi-tab interface for better user experience

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

### Resume Customization

- Ensure both your resume and job description are provided before generating customized resume
- If customization fails, try using a different AI model from the dropdown
- The AI only reorganizes existing information - it won't add fake experience or skills
- You can always switch back to your original resume using the radio buttons

### Document Format Support

#### Word Document (DOCX) Features
- **Format Detection**: The system automatically detects if your original resume is a Word document
- **Styling Preservation**: Maintains original formatting, fonts, and structure while updating content
- **Smart Output**: Customized version preserves the document's professional appearance
- **Email Attachments**: When sending emails, the system will attach the Word version if available

#### PDF Document Features
- **Advanced PDF Processing**: Uses PyMuPDF for sophisticated PDF manipulation
- **Layout Preservation**: Maintains page dimensions and basic layout from original PDF
- **Multiple Approaches**: 
  - **PyMuPDF Method**: Preserves page size and creates new styled content
  - **ReportLab Fallback**: Creates professional PDF if direct modification fails
- **Font and Spacing**: Attempts to maintain professional typography and spacing

#### Universal Features
- **Fallback Support**: If document modification fails, a text version will be provided
- **Download Options**: Multiple format download buttons appear when applicable
- **Original Preservation**: Your original file is never modified, only copied and customized
