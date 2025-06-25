#!/bin/bash

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting AI Email Assistant...${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Stop any existing containers with the same name
echo -e "${YELLOW}Stopping any existing G4F API containers...${NC}"
docker stop g4f-api 2>/dev/null || true
docker rm g4f-api 2>/dev/null || true

# Create necessary directories
mkdir -p har_and_cookies
mkdir -p generated_media

# Start the G4F API server in Docker
echo -e "${YELLOW}Starting G4F API server in Docker...${NC}"
docker run -d --name g4f-api \
  -p 1337:8080 -p 8080:8080 \
  -v "${PWD}/har_and_cookies:/app/har_and_cookies" \
  -v "${PWD}/generated_media:/app/generated_media" \
  hlohaus789/g4f:latest-slim

# Wait for the API server to start
echo -e "${YELLOW}Waiting for G4F API server to start...${NC}"
sleep 5

# Start the Streamlit app
echo -e "${GREEN}Starting Streamlit app...${NC}"
streamlit run app.py 