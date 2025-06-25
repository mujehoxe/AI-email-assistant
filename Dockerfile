FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for G4F
RUN mkdir -p /app/har_and_cookies
RUN mkdir -p /app/generated_media

# Expose ports
EXPOSE 8501 1337

# Create startup script to run both the G4F API server and Streamlit app
RUN echo '#!/bin/bash\n\
# Start G4F API server in the background\n\
python -c "from g4f.api import run_api; run_api(host=\"0.0.0.0\", port=1337)" &\n\
\n\
# Wait a moment for G4F to start\n\
sleep 5\n\
\n\
# Start Streamlit app\n\
streamlit run app.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/start.sh

RUN chmod +x /app/start.sh

# Set the entry point
ENTRYPOINT ["/app/start.sh"] 