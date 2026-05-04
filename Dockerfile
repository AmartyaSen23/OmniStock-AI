FROM python:3.10

# Create a user to avoid running as root
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy everything from your subfolder into the Docker root
COPY --chown=user CSE_274_Universal_Trader/ .

# Now it can find the requirements file we just copied over
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Hugging Face Spaces runs on port 7860 by default
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]