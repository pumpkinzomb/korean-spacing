#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Setting up environment for Korean Spacing Trainer..."

# 1️⃣ Install Node.js + npm (latest LTS)
if ! command -v node >/dev/null 2>&1; then
    echo "📦 Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get update
    apt-get install -y nodejs
else
    echo "✅ Node.js already installed: $(node -v)"
fi

# 2️⃣ Update npm (optional but nice)
npm install -g npm@latest

# 3️⃣ Install uv (if not installed)
if ! command -v uv >/dev/null 2>&1; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | bash
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "✅ uv already installed: $(uv --version)"
fi

# 4️⃣ Ensure uv in PATH for future shells
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/.local/bin:$PATH"
fi

# 5️⃣ Install project dependencies via uv (pyproject)
if [ -f "pyproject.toml" ]; then
    echo "🐍 Installing Python dependencies via uv sync..."
    uv sync
elif [ -f "requirements.txt" ]; then
    echo "🐍 Installing Python dependencies via requirements.txt..."
    uv pip install -r requirements.txt
fi

echo "🎉 Environment setup complete!"
