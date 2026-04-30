FROM python:3.11-slim

WORKDIR /app

# Install splitwise-mcp directly from GitHub (no directory copy needed)
RUN pip install --no-cache-dir git+https://github.com/tarunn2799/splitwise-mcp.git

# Install the assistant
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

ENV PORT=8000

EXPOSE 8000

CMD ["python", "-m", "splitwise_assistant.main"]
