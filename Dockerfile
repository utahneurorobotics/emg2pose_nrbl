# Use an official Python image as the base
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy only necessary files for installation
COPY environment.yml ./  
COPY setup.py setup.cfg ./

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./  
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install the package in editable mode
COPY . .  
RUN pip install -e .

# Verify the installation
RUN python -c "import emg2pose; print('emg2pose installed successfully')"

# Install JupyterLab
RUN pip install jupyterlab ipykernel

RUN pip install -e emg2pose/UmeTrack
ENV PYTHONPATH="/app/emg2pose:$PYTHONPATH"

# Expose ports
EXPOSE 8000
EXPOSE 8888

# Start Jupyter Lab
CMD ["bash", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' & bash"]