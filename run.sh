#!/bin/bash

# Agentic RAG Application with Gemini - macOS Script

echo "🚀 Starting Agentic RAG Application with Google Gemini on macOS..."

# Environment setup with options
setup_environment() {
    # Check if we're in a conda environment
    if [[ "$CONDA_DEFAULT_ENV" != "" && "$CONDA_DEFAULT_ENV" != "base" ]]; then
        echo "📦 Detected conda environment: $CONDA_DEFAULT_ENV"
        echo "🤔 Choose your setup option:"
        echo "1) Use current conda environment (recommended)"
        echo "2) Create new virtual environment"
        echo "3) Skip dependency installation (if already installed)"
        
        read -p "Enter your choice (1-3) [default: 1]: " choice
        choice=${choice:-1}
        
        case $choice in
            1)
                echo "✅ Using conda environment: $CONDA_DEFAULT_ENV"
                detect_python_command
                INSTALL_DEPS=true
                ;;
            2)
                echo "📦 Creating new virtual environment..."
                if [ ! -d "venv" ]; then
                    python3 -m venv venv
                fi
                source venv/bin/activate
                detect_python_command
                INSTALL_DEPS=true
                ;;
            3)
                echo "⏭️ Skipping dependency installation"
                detect_python_command
                INSTALL_DEPS=false
                ;;
            *)
                echo "❌ Invalid choice. Using conda environment."
                detect_python_command
                INSTALL_DEPS=true
                ;;
        esac
    else
        echo "📦 No conda environment detected. Creating virtual environment..."
        if [ ! -d "venv" ]; then
            python3 -m venv venv
        fi
        source venv/bin/activate
        detect_python_command
        INSTALL_DEPS=true
    fi
}

# Function to detect the correct Python command
detect_python_command() {
    if command -v python &> /dev/null; then
        PYTHON_CMD="python"
        echo "🐍 Using Python command: python"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        echo "🐍 Using Python command: python3"
    else
        echo "❌ No Python command found. Please ensure Python is installed."
        exit 1
    fi
    
    # Verify Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
    echo "✅ $PYTHON_VERSION detected"
}

# Setup environment
setup_environment

# Install dependencies if needed
if [ "$INSTALL_DEPS" = true ]; then
    # Check if uv is installed
    if command -v uv &> /dev/null; then
        echo "⚡ Using uv for fast package installation..."
        
        # Install with uv (much faster than pip)
        uv pip install -r requirements.txt
        echo "✅ Dependencies installed successfully with uv"
    else
        echo "📦 uv not found, using pip (consider installing uv for faster installs)"
        echo "   Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo ""
        
        echo "🔧 Upgrading pip..."
        $PYTHON_CMD -m pip install --upgrade pip
        
        echo "📥 Installing requirements with pip..."
        pip install -r requirements.txt
        echo "✅ Dependencies installed successfully with pip"
    fi
else
    echo "⏭️ Skipping dependency installation as requested"
fi

# Check environment variables
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ Error: GOOGLE_API_KEY environment variable is required"
    echo "Please set your Google API key:"
    echo "export GOOGLE_API_KEY=your_api_key_here"
    echo "Get your API key from: https://makersuite.google.com/app/apikey"
    echo ""
    echo "You can also add it to your ~/.zshrc:"
    echo "echo 'export GOOGLE_API_KEY=your_api_key_here' >> ~/.zshrc"
    echo "source ~/.zshrc"
    exit 1
fi

# Check Docker setup (Colima or Docker Desktop)
echo "🗄️ Checking Docker setup..."

# Function to start Qdrant with proper Docker setup
start_qdrant() {
    # Stop and remove existing container if it exists
    if docker ps -a --format 'table {{.Names}}' | grep -q qdrant 2>/dev/null; then
        echo "🔄 Stopping existing Qdrant container..."
        docker stop qdrant 2>/dev/null || true
        docker rm qdrant 2>/dev/null || true
    fi

    echo "🚀 Starting fresh Qdrant container..."
    docker run -d --name qdrant \
        -p 6333:6333 -p 6334:6334 \
        -v "${PWD}/qdrant_storage:/qdrant/storage" \
        qdrant/qdrant:latest

    # Wait for Qdrant to be ready
    echo "⏳ Waiting for Qdrant to start..."
    for i in {1..30}; do
        if curl -s http://localhost:6333/health > /dev/null 2>&1; then
            echo "✅ Qdrant is running and accessible"
            return 0
        fi
        sleep 1
    done
    
    echo "⚠️ Qdrant may not be ready, but continuing..."
    return 0
}

# Check if Docker command works
if ! command -v docker &> /dev/null; then
    echo "❌ Docker command not found. Please install Docker Desktop or Colima:"
    echo "Docker Desktop: https://docs.docker.com/desktop/install/mac-install/"
    echo "Or Colima: brew install colima"
    exit 1
fi

# Try to connect to Docker daemon
if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running."
    echo "Detected Docker socket: $DOCKER_HOST"
    
    # Check if using Colima
    if [[ "$DOCKER_HOST" == *"colima"* ]] || [[ -S "$HOME/.colima/default/docker.sock" ]]; then
        echo "📦 Detected Colima setup. Starting Colima..."
        if command -v colima &> /dev/null; then
            colima start
            echo "✅ Colima started"
        else
            echo "❌ Colima not found. Install with: brew install colima"
            exit 1
        fi
    else
        echo "📦 Please start Docker Desktop or run: colima start"
        exit 1
    fi
fi

# Verify Docker is working
if docker info &> /dev/null; then
    echo "✅ Docker is running"
    start_qdrant
else
    echo "❌ Could not connect to Docker. Please check your Docker setup."
    exit 1
fi

# Create necessary directories
mkdir -p assets/sample_docs
mkdir -p temp

# Set Python path
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Run the application
echo "🤖 Starting Gradio interface with Gemini..."
echo "📱 Open http://localhost:7860 in your browser"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

# Navigate to src directory and run the application
cd src && $PYTHON_CMD gradio_demo.py