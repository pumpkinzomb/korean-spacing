#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
fi

echo "🚀 Setting up environment for Korean Spacing Trainer..."

ensure_node_tooling() {
    if command -v npm >/dev/null 2>&1; then
        echo "✅ npm already installed: $(npm -v)"
        return
    fi

    if ! command -v apt-get >/dev/null 2>&1; then
        echo "❌ npm not found and automatic installation requires apt-get." >&2
        echo "   Please install Node.js 20+ / npm manually and re-run." >&2
        exit 1
    fi

    echo "📦 Installing Node.js + npm via NodeSource..."
    TMP_SCRIPT="$(mktemp)"
    curl -fsSL https://deb.nodesource.com/setup_20.x -o "$TMP_SCRIPT"
    $SUDO bash "$TMP_SCRIPT"
    rm -f "$TMP_SCRIPT"

    $SUDO apt-get update
    $SUDO apt-get install -y nodejs

    echo "✅ npm installed: $(npm -v)"
}

ensure_node_tooling
npm install -g npm@latest

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        echo "✅ uv already installed: $(uv --version)"
        return
    fi

    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
}

ensure_uv

if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ -f "pyproject.toml" ]; then
    echo "🐍 Installing Python dependencies via uv sync..."
    uv sync
elif [ -f "requirements.txt" ]; then
    echo "🐍 Installing Python dependencies via requirements.txt..."
    uv pip install -r requirements.txt
fi

echo "🎉 Environment setup complete!"
