# Email Sender for Recruiters

A Streamlit application to send customized emails to recruiters with resume attachments and AI-powered template generation.

## Features

- Automatically extracts recruiter's name from email address
- Editable email template with separate greeting, body, and signature sections
- Resume attachment option
- Configurable SMTP settings for any email provider
- AI-powered template generation based on job descriptions and your resume
- Template management system with save, load, and delete options
- User settings for custom email credentials and personalization

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```

## Configuration

The application provides two ways to configure your email settings:

1. **Settings Sidebar**: Use the Settings section in the sidebar to enter your email, password, SMTP server, and name
2. **Environment Variables**: Create a `.env` file with the following variables:
   ```
   EMAIL_ADDRESS=your_email@example.com
   EMAIL_PASSWORD=your_password
   ```

For security, we recommend using an app password instead of your regular password, especially if you have two-factor authentication enabled.

### SMTP Configuration

The app comes with default settings for Outlook, but you can configure it to work with any email provider:

- **Outlook**: smtp-mail.outlook.com:587
- **Gmail**: smtp.gmail.com:587
- **Yahoo**: smtp.mail.yahoo.com:587

The application will automatically try to infer the correct SMTP settings based on your email domain.

## Getting an App Password

If you have two-factor authentication enabled on your email account, you'll need to generate an app password:

1. Go to your email provider's security settings
2. Enable 2-factor authentication if not already enabled
3. Generate an app password
4. Use this app password in the Settings sidebar

## Usage

### Basic Email Sending

1. Enter the recipient's email address
2. The app will automatically detect the recipient's name or you can specify it manually
3. Enter the position you're applying for
4. Customize the email subject, greeting, body, and signature
5. Upload your resume
6. Click "Send Email"

### AI Template Generation

1. Upload your resume (PDF or Word format)
2. Click on the "AI Template Generator" expander
3. Paste the job description
4. Click "Generate AI Template"
5. The AI will analyze your resume and the job description to create a tailored email template
6. The generated template will be automatically saved and can be modified before sending

### Personalizing Your Templates

1. Go to the Settings section in the sidebar
2. Enter your name, which will appear in the signature of your emails
3. Your templates will be updated with your name in the signature

## Deployment

### Docker Deployment

You can run this application using Docker:

```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Render Deployment

This application can be deployed to Render using the provided `render.yaml` file.

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).
