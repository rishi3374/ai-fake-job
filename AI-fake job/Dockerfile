# Multi-stage Dockerfile for AI Fake Job Detector
FROM python:3.12-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage for backend
FROM base as backend

# Copy backend code
COPY backend/ ./backend/
COPY models/ ./models/
COPY preprocessing/ ./preprocessing/
COPY explainability/ ./explainability/
COPY ocr/ ./ocr/
COPY verification/ ./verification/
COPY anomaly/ ./anomaly/
COPY data/ ./data/

# Set environment variables
ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "backend/api/main.py"]

# Stage for frontend
FROM node:18-alpine as frontend

WORKDIR /app/frontend

# Copy frontend files
COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./

# Build frontend
RUN npm run build

# Expose port
EXPOSE 3000

# Start frontend
CMD ["npm", "run", "dev"]

# Production stage
FROM nginx:alpine as production

# Copy frontend build
COPY --from=frontend /app/frontend/dist /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
