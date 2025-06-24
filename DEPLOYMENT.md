# Deployment Guide

This guide explains how to deploy the Email Sender application using Docker and Render.

## Local Deployment with Docker

### Prerequisites

- Docker and Docker Compose installed on your machine

### Steps

1. Clone the repository:

   ```
   git clone <repository-url>
   cd email-sender
   ```

2. (Optional) Create a `.env` file with your email credentials:

   ```
   EMAIL_ADDRESS=your_email@example.com
   EMAIL_PASSWORD=your_app_password
   ```

3. Build and run the Docker container:

   ```
   docker-compose up --build
   ```

4. Access the application at: http://localhost:8501

## Deployment on Render

### Prerequisites

- A Render account (https://render.com)
- Git repository hosted on GitHub, GitLab, or Bitbucket

### Steps

1. Fork or push the repository to your Git hosting service.

2. Log in to your Render account.

3. Click on "New" and select "Blueprint" from the dropdown menu.

4. Connect your repository and select it.

5. Render will automatically detect the `render.yaml` file and configure the service.

6. Set the following environment variables in the Render dashboard:

   - `EMAIL_ADDRESS`: Your email address (optional)
   - `EMAIL_PASSWORD`: Your email password or app password (optional)

7. Click "Create Blueprint" and wait for the deployment to complete.

8. Once deployed, you can access your application at the URL provided by Render.

## Important Notes

- For security, we recommend using app passwords instead of your regular account password, especially if you have two-factor authentication enabled.
- The application will automatically try to infer SMTP settings from the email domain, but you can manually configure them in the settings.
- If you're deploying to Render, note that the free tier may have limitations on uptime and performance.

## Troubleshooting

- If you encounter issues with the G4F API, try restarting the application or check for updates to the g4f package.
- For SMTP connection issues, ensure your email provider allows SMTP access and that you're using the correct server and port.
- If you're using Gmail, you may need to enable "Less secure app access" or use an app password.
