FROM python:3.11-slim

WORKDIR /app

# Install the splitwise-mcp dependency
COPY splitwise-mcp/ /app/splitwise-mcp/
RUN pip install --no-cache-dir -e /app/splitwise-mcp

# Install the assistant
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

ENV SPLITWISE_MCP_PATH=/app/splitwise-mcp/app.py
ENV PORT=8000

EXPOSE 8000

CMD ["python", "-m", "splitwise_assistant.main"]
