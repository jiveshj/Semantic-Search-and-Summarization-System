 ------------------------------------------------------------------------------
# Docker Configuration Files
# ------------------------------------------------------------------------------

# Dockerfile
DOCKERFILE_CONTENT = """
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Copy application code
COPY . .

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/indexes

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "api_service.py"]
"""

# docker-compose.yml
DOCKER_COMPOSE_CONTENT = """
version: '3.8'

services:
  semantic-search:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - INDEX_PATH=/app/indexes/search_index
      - PORT=8000
    volumes:
      - ./data:/app/data
      - ./indexes:/app/indexes
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Optional: Add Redis for improved caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - ./redis-data:/data
    restart: unless-stopped

  # Optional: Add monitoring with Prometheus
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

networks:
  default:
    driver: bridge
"""

# requirements.txt
REQUIREMENTS_CONTENT = """
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
numpy==1.24.3
pandas==2.0.3
torch==2.1.0
transformers==4.35.2
sentence-transformers==2.2.2
faiss-cpu==1.7.4
scikit-learn==1.3.2
nltk==3.8.1
requests==2.31.0
beautifulsoup4==4.12.2
PyPDF2==3.0.1
python-docx==1.1.0
python-multipart==0.0.6
aiohttp==3.9.1
tqdm==4.66.1
"""

# kubernetes/deployment.yaml
KUBERNETES_DEPLOYMENT = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: semantic-search-deployment
  labels:
    app: semantic-search
spec:
  replicas: 3
  selector:
    matchLabels:
      app: semantic-search
  template:
    metadata:
      labels:
        app: semantic-search
    spec:
      containers:
      - name: semantic-search
        image: semantic-search:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: INDEX_PATH
          value: "/app/indexes/search_index"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: index-volume
          mountPath: /app/indexes
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: semantic-search-data-pvc
      - name: index-volume
        persistentVolumeClaim:
          claimName: semantic-search-index-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: semantic-search-service
spec:
  selector:
    app: semantic-search
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: semantic-search-data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: semantic-search-index-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
"""

# client_example.py - Example client usage
CLIENT_EXAMPLE = """
import requests
import json

# Example client for the Semantic Search API
class SemanticSearchClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def search(self, query: str, k: int = 10, include_summary: bool = True):
        response = requests.post(
            f"{self.base_url}/search",
            json={
                "query": query,
                "k": k,
                "include_summary": include_summary
            }
        )
        return response.json()
    
    def add_document(self, title: str, content: str, source: str = "client"):
        response = requests.post(
            f"{self.base_url}/documents",
            json={
                "title": title,
                "content": content,
                "source": source
            }
        )
        return response.json()
    
    def get_stats(self):
        response = requests.get(f"{self.base_url}/stats")
        return response.json()
    
    def health_check(self):
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Example usage
if __name__ == "__main__":
    client = SemanticSearchClient()
    
    # Add a document
    result = client.add_document(
        title="AI Ethics",
        content="Artificial Intelligence ethics involves ensuring AI systems are fair, transparent, and beneficial to society..."
    )
    print("Document added:", result)
    
    # Search for documents
    results = client.search("What is artificial intelligence ethics?")
    
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Title: {result['title']}")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Summary: {result['summary']}")
        print()
"""

print("📁 Additional Files Created:")
print("- Dockerfile")
print("- docker-compose.yml") 
print("- requirements.txt")
print("- kubernetes/deployment.yaml")
print("- client_example.py")
print("\nThese files provide a complete production-ready deployment setup!")
