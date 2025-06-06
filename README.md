# Embedding Inversion Demo

This repo contains Jupyter notebooks showcasing inversion embedding attacks on ChromaDB & CyborgDB. More DBs may be added later.

## Quickstart

1. Set up Conda environment

```sh
mamba env create -f environment.yml
```

2. Install & launch Redis server (for CyborgDB)

```sh
# macOS
brew install redis
brew services start redis

# Ubuntu/Debian  
sudo apt install redis-server
sudo systemctl start redis

# Docker
docker run -d -p 6379:6379 redis:alpine
```

3. Set your OpenAI API Key (for embedding generation)

```sh
export OPENAI_API_KEY="sk..."
```

4. Run the notebooks!