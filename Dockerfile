FROM python:3.11
COPY . /project/
WORKDIR /project

# Install dependencies
RUN apt-get update && \
    apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/*

# Download and install Gurobi
RUN wget https://packages.gurobi.com/12.0/gurobi12.0.0_linux64.tar.gz && \
    tar xvf gurobi12.0.0_linux64.tar.gz && \
    rm gurobi12.0.0_linux64.tar.gz && \
    mv gurobi1200 /opt/gurobi

# Set environment variables
ENV PATH=/opt/gurobi/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/gurobi/lib:${LD_LIBRARY_PATH} \
    GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic

# Install Python dependencies
RUN pip install -e .