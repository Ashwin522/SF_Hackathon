#!/bin/bash
# GitHub Setup Script for Basketball Tactical Analysis

echo "==========================================="
echo "Basketball Tactical Analysis - Git Setup"
echo "==========================================="
echo ""

# Navigate to project directory
cd /Users/ashwinnair/Desktop/final_try_basketball/openai_gym

# Initialize git if not already initialized
if [ ! -d .git ]; then
    echo "Initializing Git repository..."
    git init
    echo "✓ Git initialized"
else
    echo "✓ Git already initialized"
fi

# Add all files
echo ""
echo "Staging files..."
git add .

# Create initial commit
echo ""
echo "Creating initial commit..."
git commit -m "Initial commit: Basketball tactical analysis system with LLM integration"

echo ""
echo "==========================================="
echo "✓ Local Git repository ready!"
echo "==========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   https://github.com/new"
echo ""
echo "2. Name it: basketball-tactical-analysis"
echo ""
echo "3. DO NOT initialize with README, .gitignore, or license"
echo ""
echo "4. Run these commands:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/basketball-tactical-analysis.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "==========================================="
echo ""
echo "⚠️  IMPORTANT: Your API key will NOT be uploaded (it's in .gitignore)"
echo "    Users will need to set their own API keys using .env.example"
echo ""
