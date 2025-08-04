#!/usr/bin/env python3
"""
Quick start script for the Agri Commodities Sentiment Dashboard
This script will start both the backend API and frontend development server
"""

import os
import sys
import subprocess
import time
import threading
from pathlib import Path

def print_banner():
    print("\n" + "="*70)
    print("🌾 AGRI COMMODITIES SENTIMENT DASHBOARD")
    print("="*70)
    print("Starting both backend API and frontend development server...")
    print()

def check_requirements():
    """Check if Python and Node.js are available"""
    try:
        # Check Python
        python_version = subprocess.check_output([sys.executable, "--version"], 
                                               stderr=subprocess.STDOUT, text=True)
        print(f"✅ Python: {python_version.strip()}")
        
        # Check Node.js
        node_version = subprocess.check_output(["node", "--version"], 
                                             stderr=subprocess.STDOUT, text=True)
        print(f"✅ Node.js: {node_version.strip()}")
        
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Missing requirement: {e}")
        return False

def start_backend():
    """Start the FastAPI backend server"""
    print("\n🚀 Starting Backend Server...")
    print("   📍 API: http://localhost:8000")
    print("   📚 Docs: http://localhost:8000/docs")
    
    try:
        # Install backend dependencies if needed
        print("   📦 Installing backend dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        
        # Start the backend server directly with uvicorn
        subprocess.run([sys.executable, "-m", "uvicorn", "backend_api:app", 
                       "--host", "0.0.0.0", "--port", "8000", "--reload"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start backend: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Backend server stopped by user")
        return True
    return True

def start_frontend():
    """Start the Next.js frontend development server"""
    print("\n🎨 Starting Frontend Server...")
    print("   📍 Web App: http://localhost:3000")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found!")
        return False
    
    try:
        # Change to frontend directory
        os.chdir(frontend_dir)
        
        # Install dependencies if node_modules doesn't exist
        if not Path("node_modules").exists():
            print("   📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], check=True)
        
        # Start the development server
        subprocess.run(["npm", "run", "dev"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start frontend: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Frontend server stopped by user")
        return True
    return True

def run_backend_thread():
    """Run backend in a separate thread"""
    start_backend()

def main():
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please install Python 3.8+ and Node.js 18+")
        sys.exit(1)
    
    # Check for .env file
    if not Path(".env").exists():
        print("\n⚠️  .env file not found!")
        print("   Please copy .env.template to .env and configure your settings")
        if input("   Continue anyway? (y/N): ").lower() != 'y':
            sys.exit(1)
    
    print("\n🔧 Setup complete! Starting servers...")
    print("   Press Ctrl+C to stop both servers")
    print("="*70)
    
    try:
        # Start backend in a separate thread
        backend_thread = threading.Thread(target=run_backend_thread, daemon=True)
        backend_thread.start()
        
        # Give backend time to start
        time.sleep(3)
        
        # Start frontend in main thread (so Ctrl+C works properly)
        start_frontend()
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down servers...")
        print("Thank you for using Agri Commodities Sentiment Dashboard!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()