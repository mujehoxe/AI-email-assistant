# AI Email Assistant

An AI-powered tool to create personalized job application emails using AI models from G4F.

## Features

- AI-generated email templates based on job descriptions
- Dynamic model selection from available G4F models
- Resume parsing and analysis
- Email configuration with automatic SMTP detection
- Clipboard functionality for easy template copying
- Docker and Render deployment support

## Local Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your email settings:
   ```
   EMAIL_ADDRESS=your.email@example.com
   EMAIL_PASSWORD=your-email-password
   ```
4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
5. Run the G4F API server (in a separate terminal):
   ```
   python -c "from g4f.api import run_api; run_api()"
   ```

## Docker Setup

You can also run the application using Docker:

```bash
docker build -t email-ai-assistant .
docker run -p 8501:8501 -p 1337:1337 email-ai-assistant
```

Alternatively, you can use the prebuilt G4F image:

```bash
docker run \
  -p 1337:8080 -p 8080:8080 \
  -v ${PWD}/har_and_cookies:/app/har_and_cookies \
  -v ${PWD}/generated_media:/app/generated_media \
  hlohaus789/g4f:latest-slim
```

And then run the Streamlit app separately:

```bash
streamlit run app.py
```

### Troubleshooting Docker

If the G4F API server is not working properly, try these steps:

1. Ensure ports 1337 and 8080 are available and not being used by other services
2. Make sure Docker has permission to create and access the mounted volumes
3. Check Docker logs for any error messages:
   ```bash
   docker ps -a  # Get container ID
   docker logs <container_id>
   ```
4. If issues persist, try restarting the Docker daemon:
   ```bash
   sudo systemctl restart docker
   ```
5. Verify the API server is running by accessing: http://localhost:1337/models

## Deployment on Render

This application can be deployed to Render using the included configuration:

1. Fork this repository to your GitHub account
2. Sign up for a [Render](https://render.com/) account
3. Create a new Web Service in Render
4. Connect your GitHub repository
5. Select "Docker" as the environment
6. Set the following environment variables:
   - G4F_API_HOST=localhost
   - G4F_API_PORT=1337
7. Click "Create Web Service"

The application will be deployed and both the Streamlit app and G4F API server will run in the same container.

## Configuration

Email settings can be configured in the sidebar. The application will attempt to automatically detect the SMTP server based on your email domain.

## Available Models

The application fetches available models from the G4F API server. If the server is not running or cannot be reached, a list of fallback models will be used instead.
