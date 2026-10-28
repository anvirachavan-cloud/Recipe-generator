"""
Application startup script — installs dependencies and launches the Flask server.
Run: python run.py
"""

import subprocess
import sys
import os


def install_requirements():
    print("📦 Installing dependencies from requirements.txt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])
    print("✅ Dependencies installed.\n")


def check_documents():
    if not os.path.exists("documents"):
        os.makedirs("documents")
        print("📁 Created documents/ directory. Add your culinary PDFs, TXT, or DOCX files there.")
    else:
        files = [f for f in os.listdir("documents") if f.endswith((".txt", ".pdf", ".docx"))]
        print(f"📚 Found {len(files)} document(s) in documents/: {', '.join(files) if files else 'none'}")


def main():
    print("=" * 60)
    print("  🍳 Chef Granite — Recipe Generator RAG Agent")
    print("  Powered by IBM Granite · watsonx.ai")
    print("=" * 60)
    print()

    install_requirements()
    check_documents()

    print("\n🚀 Starting Flask server on http://localhost:5000")
    print("   Press CTRL+C to stop.\n")

    os.environ.setdefault("FLASK_APP", "app.py")
    os.environ.setdefault("PORT", "5000")

    from app import app
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
