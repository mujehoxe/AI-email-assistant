# Start with the G4F image
FROM hlohaus789/g4f:latest-slim

# Set working directory
WORKDIR /app

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p data har_and_cookies generated_media

# Set environment variables for port and G4F API
ENV PORT=8501
ENV G4F_API_HOST=127.0.0.1
ENV G4F_API_PORT=8080

# Create start script
RUN echo '#!/bin/bash\n\
echo "Starting G4F API server..."\n\
python -m g4f.api.server --host 127.0.0.1 --port 8080 &\n\
G4F_PID=$!\n\
echo "G4F API server started with PID: $G4F_PID"\n\
\n\
# Wait for G4F API to start\n\
echo "Waiting for G4F API to be ready..."\n\
sleep 5\n\
\n\
# Start Streamlit\n\
echo "Starting Streamlit app..."\n\
streamlit run app.py --server.port $PORT --server.address 0.0.0.0\n\
' > /app/start.sh

# Make start script executable
RUN chmod +x /app/start.sh

# Start the application
CMD ["/app/start.sh"] 