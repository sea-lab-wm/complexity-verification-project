# =================================
# Miniconda3
# =================================

FROM continuumio/miniconda3

ARG ENV_NAME

SHELL ["/bin/bash","-l", "-c"]

WORKDIR /root/

# Install Miniconda
RUN /opt/conda/bin/conda init bash && \
    /opt/conda/bin/conda config --add channels jolespin && \
    /opt/conda/bin/conda config --add channels bioconda && \
    /opt/conda/bin/conda config --add channels conda-forge && \
    /opt/conda/bin/conda update conda -y && \
    /opt/conda/bin/conda install -c conda-forge mamba -y && \
    /opt/conda/bin/conda clean -afy

# =================================

# Add conda bin to path
ENV PATH /opt/conda/bin:$PATH

# Create environment
RUN conda create -n ${ENV_NAME} python=3.11 -y

# Set up environment
RUN echo "conda activate ${ENV_NAME}" >> ~/.bashrc

# install packages from environment.yml
COPY environment.yml .

RUN conda env update -n ${ENV_NAME} -f environment.yml && \
    conda clean -afy

# workdir
WORKDIR /verification_project

# Set entrypoint to bash
ENTRYPOINT ["bash", "-l", "-c"]