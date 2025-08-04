#!/usr/bin/env python3
"""
Script to run the FastAPI backend server for the Agri Commodities Sentiment Dashboard
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check if all required environment variables are set"""
    required_vars = [
        "DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_PORT"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these environment variables in your .env file or system environment.")
        return False
    
    print("✅ All required environment variables are set")
    return True

def install_dependencies():
    """Install required Python dependencies"""
    dependencies = [
        "fastapi",
        "uvicorn",
        "psycopg2-binary",
        "pandas",
        "python-dotenv",
        "langchain",
        "langchain-openai",
        "langchain-community"
    ]
    
    print("📦 Installing Python dependencies...")
    import subprocess
    
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", dep])
            print(f"   ✅ {dep}")
        except subprocess.CalledProcessError:
            print(f"   ❌ Failed to install {dep}")
            return False
    
    return True

def main():
    print("🚀 Starting Agri Commodities Sentiment Dashboard Backend")
    print("=" * 60)
    
    # Check environment variables
    if not check_environment():
        sys.exit(1)
    
    # Check for OpenAI API Key (optional for basic functionality)
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found. Chat functionality will be limited.")
    else:
        print("✅ OpenAI API Key found")
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    print("\n🌟 Starting FastAPI server...")
    print("   📍 API will be available at: http://localhost:8001")
    print("   📚 API documentation: http://localhost:8001/docs")
    print("   🔄 Health check: http://localhost:8001/health")
    print("\n   Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Start the server
    try:
        import uvicorn
        host = os.environ.get("API_HOST", "127.0.0.1")
        port = int(os.environ.get("API_PORT", 8001))
        uvicorn.run("backend_api:app", host=host, port=port, reload=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()