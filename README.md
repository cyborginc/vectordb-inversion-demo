# Embedding Inversion Demo

This repo contains Jupyter notebooks showcasing inversion embedding attacks on ChromaDB & CyborgDB. More DBs may be added later.

## Quickstart

1. Set up Conda environment

```sh
mamba env create -f environment.yml
```

2. Install & launch PostgreSQL server (for CyborgDB)

```sh
# macOS
brew install postgresql
brew services start postgresql

# Ubuntu/Debian  
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Docker
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:alpine
```

3. Set your OpenAI API Key (for embedding generation)

```sh
export OPENAI_API_KEY="sk..."
```

4. Update your Postgres credentials in the [`cyborgdb-vec2text.ipynb`](cyborgdb-vec2text.ipynb) notebook

```py
# Set up PostgreSQL connection parameters
# Make sure to change these to your actual PostgreSQL credentials
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432
POSTGRES_DB = "postgres"
POSTGRES_USER = "nicolas"
POSTGRES_PASSWORD = "password"
```

4. Run the notebooks!