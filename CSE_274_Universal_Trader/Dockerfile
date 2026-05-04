FROM python:3.10

# Create a user to avoid running as root
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy and install requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the code
COPY --chown=user . .

# Hugging Face Spaces runs on port 7860 by default
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]