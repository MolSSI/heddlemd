FROM docker.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get clean && \
    apt-get update && \
    apt-get install -y software-properties-common && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
                       clang \
                       clangd \
                       cmake \
                       curl \
                       gcc \
                       gdb \
                       git \
                       make \
                       npm \
                       python3-venv \
                       python3-pip \
                       vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add python
RUN apt-get clean && \
    apt-get update && \
    apt-get install -y \
                       python3 \
                       python3-pip \
                       python3-dev \
                       && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Symlink python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install PyCUDA and other Python packages
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
                          pycuda \
                          numpy \
                          scipy \
                          matplotlib \
                          nvidia-cutlass-dsl==4.2.0 \
                          pandas \
                          jupyter



ENV PATH="$PATH:/root/.local/bin"

# Install jq, which is needed for the requirements index script
RUN apt-get update && \
    apt install -y jq

# Install Claude
RUN curl -fsSL https://claude.ai/install.sh | bash


# Install Rust
ENV RUST_VERSION=1.93.0
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain ${RUST_VERSION}
ENV PATH="/root/.cargo/bin:${PATH}"

# Install mdBook for building user-facing documentation
RUN cargo install mdbook --locked



COPY .podman/interface.sh /interface.sh
RUN mkdir -p /.podman
COPY .podman/interface.sh /.podman/interface.sh

# Copy the entrypoint files into the Docker image
COPY .podman/entrypoint.sh /.term/entrypoint.sh
RUN chmod +x /.term/entrypoint.sh

# Set the default command
CMD ["/.term/entrypoint.sh"]

