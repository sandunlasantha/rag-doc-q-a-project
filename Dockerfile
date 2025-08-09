# Dockerfile

# --- Stage 1: Base Image ---
# We start with an official, lightweight Python image.
FROM python:3.11-slim

# --- Stage 2: Apply Security Patches & Install Tools ---
# This is a best practice for production images.
RUN apt-get update && apt-get upgrade -y && apt-get install -y build-essential

# --- Stage 3: Set up the Environment ---
# Set the working directory inside the container.
WORKDIR /app

# --- Stage 4: Install Dependencies ---
# Copy the requirements file.
COPY requirements.txt .

# Install all the Python libraries to their standard, system-wide locations.
# This ensures that executables like 'uvicorn' are correctly placed in the system's PATH.
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 5: Copy Application Files ---
# We explicitly copy ONLY the files needed for the API to run.
COPY api.py .
COPY prompt.py . 
COPY google-2023-environmental-report.pdf .
COPY chroma_db_local/ ./chroma_db_local/

# --- Stage 6: Expose the Port ---
# Tell Docker that our application will be listening on a port.
EXPOSE 8000

# --- Stage 7: Run the Application ---
# This command works perfectly both locally and in Google Cloud Run.
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
