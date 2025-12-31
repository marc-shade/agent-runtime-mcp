#!/bin/bash
# Agent Runtime MCP Installation Script

set -e

echo "Installing Agent Runtime MCP..."

# Check if we're in the right directory
if [ ! -f "install.sh" ]; then
    echo "Error: Run this script from the agent-runtime-mcp directory"
    exit 1
fi

# Get the MCP server files from the source
echo ""
echo "⚠️  Manual Setup Required"
echo "========================================"
echo ""
echo "The Agent Runtime MCP server requires full source files."
echo ""
echo "To complete installation, you have two options:"
echo ""
echo "1. Copy from an existing node:"
echo "   scp -r source-node:/path/to/mcp-servers/agent-runtime-mcp/* ."
echo ""
echo "2. Download from GitHub (if available):"
echo "   git clone https://github.com/marc-shade/agent-runtime-mcp.git ."
echo ""
echo "Once you have the files, install dependencies:"
echo "   pip3 install -r requirements.txt"
echo ""
echo "Then test the server:"
echo "   python3 server.py"
echo ""

# Create a placeholder requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    cat > requirements.txt <<EOF
# Agent Runtime MCP Dependencies
mcp>=0.9.0
jsonschema>=4.20.0
sqlite3
asyncio>=3.4.3
structlog>=24.1.0
EOF
    echo "Created requirements.txt"
fi

echo "Installation script complete."
echo "Remember to copy the full server code before using!"
