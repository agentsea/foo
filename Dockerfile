FROM ghcr.io/astral-sh/uv:python3.11-bookworm

COPY . /app
WORKDIR /app

# Install system dependencies needed for the app or kubectl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    apt-transport-https \
    ca-certificates \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl \
    && rm kubectl

RUN curl https://rclone.org/install.sh | bash

# Copy dependency definition files
COPY pyproject.toml uv.lock* /app/

# Install Python dependencies using uv
RUN uv sync --frozen --prerelease=allow

# Copy the rest of the application code
COPY . /app/

EXPOSE 9090
ENV DB_TYPE=sqlite

# Run the application
CMD ["uv", "run", "--prerelease=allow", "python", "-m", "foo.server"]
