
FROM pfeiffermax/python-poetry:1.12.0-poetry1.8.4-python3.11.10-bookworm

COPY . /app
WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    unzip \
    apt-transport-https \
    ca-certificates \
    gnupg \
    libssl-dev \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && export PATH="$HOME/.cargo/bin:$PATH"
ENV PATH="/root/.cargo/bin:${PATH}"

RUN poetry install

EXPOSE 9090
ENV DB_TYPE=sqlite

# Install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl \
    && rm kubectl

# Run the application
CMD ["poetry", "run", "python", "-m", "foo.server"]
