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

# Set environment variable for port
ENV PORT=8501

# Start both G4F server and Streamlit
CMD sh -c "python -m g4f.api.server & streamlit run app.py --server.port $PORT --server.address 0.0.0.0" 