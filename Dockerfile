FROM python:3.11-slim

WORKDIR /app

# Install splitwise-mcp from GitHub archive (no git binary required)
RUN pip install --no-cache-dir https://github.com/tarunn2799/splitwise-mcp/archive/refs/heads/main.tar.gz

# Install the assistant
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

ENV PORT=8000

EXPOSE 8000

CMD ["python", "-m", "splitwise_assistant.main"]
