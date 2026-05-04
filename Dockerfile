FROM python:3.10

# Create a user to avoid running as root
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy everything from your subfolder into the Docker root
COPY --chown=user CSE_274_Universal_Trader/ .

# Force install uvicorn and fastapi alongside your requirements just to be safe
RUN pip install --no-cache-dir --upgrade uvicorn fastapi -r requirements.txt

# Use the 'python -m' trick to completely bypass any hidden PATH bugs
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]