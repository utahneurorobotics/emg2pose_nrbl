# Use an official Python image as the base
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy only necessary files for installation
COPY environment.yml ./
COPY setup.py setup.cfg ./

# Install dependencies using Conda
RUN apt-get update && apt-get install -y curl && \
    curl -o ~/miniconda.sh -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda env create -f environment.yml && \
    /opt/conda/bin/conda clean -afy

# Activate the environment and install the package
ENV PATH="/opt/conda/bin:$PATH"
RUN echo "source activate emg2pose" > ~/.bashrc
RUN pip install -e .

# Install JupyterLab and ipykernel in the emg2pose environment
RUN /opt/conda/bin/conda install -n emg2pose -y jupyterlab ipykernel
RUN /opt/conda/bin/conda clean -afy

# Register the Conda environment as a Jupyter kernel
RUN /opt/conda/bin/conda run -n emg2pose python -m ipykernel install --user --name emg2pose --display-name "Python (emg2pose)"


# Copy the rest of the application code
COPY . .

# Expose a port (if your application serves on a specific port)
EXPOSE 8000
EXPOSE 8888
CMD ["bash", "-c", "/opt/conda/bin/conda run -n emg2pose jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' & bash"]

# Define the default command
#CMD ["python", "-m", "emg2pose.train"] 
# this should be the entry point of your application