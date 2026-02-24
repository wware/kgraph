FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY examples.yaml .

# App listens on 8002 (see app.py docstring)
EXPOSE 8002

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8002"]
