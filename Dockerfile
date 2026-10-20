# opencode-gateway — Python Flask server
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py bridge.py ./

ENV PORT=8080 HOST=0.0.0.0
EXPOSE 8080

# Optional runtime env: ZEN_API_KEY, GATEWAY_KEY, OC_UA_VERSION
CMD ["sh", "-c", "exec python server.py"]
