#!/usr/bin/env python3
"""
Setup script for Chapter Quality Evaluation Framework
"""

import os
import sys

def create_directories():
    """Create necessary directories."""
    directories = ['data', 'data/extracted_chapters', 'data/outputs', 'data/manual_reviews', 'docs', 'examples', 'src']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory already exists: {directory}")

def check_requirements():
    """Check if requirements.txt exists."""
    if os.path.exists('requirements.txt'):
        print("✓ requirements.txt found")
        print("  Run: pip install -r requirements.txt")
    else:
        print("❌ requirements.txt not found")

def check_env_file():
    """Check for .env file."""
    if os.path.exists('.env'):
        print("✓ .env file found")
    else:
        print("⚠️ .env file not found")
        print("  Create .env file with: OPENAI_API_KEY=your_key_here")

def main():
    """Main setup function."""
    print("Chapter Quality Evaluation Framework Setup")
    print("=" * 50)
    
    create_directories()
    print()
    check_requirements()
    check_env_file()
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Set up .env file with your OpenAI API key")
    print("3. Run: python evaluator.py <video_id>")

if __name__ == "__main__":
    main()
