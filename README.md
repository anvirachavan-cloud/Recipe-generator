# 🍳 Chef Granite — AI Recipe Generator RAG Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0.3-lightgrey?logo=flask&logoColor=white)
![IBM watsonx](https://img.shields.io/badge/IBM%20watsonx.ai-Granite%20LLM-0530ad?logo=ibm&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

**A Document Q&A RAG (Retrieval-Augmented Generation) Agent for personalized recipe generation and culinary Q&A, powered by IBM Granite LLM on IBM watsonx.ai.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [API Reference](#-api-reference)
- [Adding Your Own Documents](#-adding-your-own-documents)
- [Guardrails & AI Ethics](#-guardrails--ai-ethics)
- [Tech Stack](#-tech-stack)
- [Disclaimer](#-disclaimer)

---

## 🌟 Overview

Chef Granite is an **Agentic AI application** that:

- **Ingests** culinary documents — cookbooks, dietary guidelines, food safety manuals, and global cuisine guides — into a semantic vector store
- **Collects** user preferences: diet type, available ingredients, allergy restrictions, and cooking constraints
- **Retrieves** relevant context using vector similarity search (RAG pattern)
- **Generates** personalized, safe recipes and culinary answers using IBM Granite LLM via IBM watsonx.ai
- **Applies** multi-layer guardrails for food safety, allergen detection, and AI ethics

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🍽️ **Recipe Generator** | Personalized recipes based on diet, ingredients, allergies, and constraints |
| 💬 **Culinary Q&A Chat** | RAG-enhanced answers to any cooking, nutrition, or food safety question |
| 🛡️ **Multi-Layer Guardrails** | Input validation, dietary compliance, allergen detection, food safety filters |
| 📚 **Document RAG** | Retrieves grounded answers from ingested culinary knowledge base |
| 🔄 **Hot Reload** | Add new documents and rebuild the vector store without restarting |
| 🌐 **Web UI** | Clean dark-theme single-page interface |
| ⚡ **REST API** | JSON API for integration with external systems |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Browser (UI)                        │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP
┌─────────────────────▼───────────────────────────────────────┐
│              Flask Web Application  (app.py)                 │
│   /api/recipe/generate  │  /api/qa  │  /api/agent/info       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│               Recipe RAG Agent  (rag_agent.py)               │
│                                                              │
│  ┌───────────────┐    ┌──────────────────┐    ┌──────────┐  │
│  │  Guardrails   │    │  Vector Retrieval │    │  Prompt  │  │
│  │  (guardrails) │───▶│  (ChromaDB)       │───▶│ Builder  │  │
│  └───────────────┘    └──────────────────┘    └────┬─────┘  │
│                                                     │        │
│  ┌──────────────────────────────────────────────────▼─────┐  │
│  │         IBM watsonx.ai  — IBM Granite / Llama 3.3      │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│        Document Ingestion Pipeline  (document_ingestion.py)  │
│  PDF / TXT / DOCX → Chunks → Embeddings → ChromaDB          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Recipe Generator/
├── app.py                          # Flask web application & REST API
├── rag_agent.py                    # Core RAG Agent orchestration
├── granite_llm.py                  # IBM watsonx.ai / Granite LLM integration
├── document_ingestion.py           # Document loading, chunking & vector store
├── guardrails.py                   # AI ethics, safety & allergen guardrails
├── run.py                          # Startup helper script
├── requirements.txt                # Python dependencies
├── config.env                      # Environment variables (API keys) — not committed
├── documents/                      # 📚 Culinary knowledge base documents
│   ├── cookbook_classic_recipes.txt
│   ├── dietary_guidelines_nutrition.txt
│   ├── food_safety_allergen_guidelines.txt
│   └── global_cuisine_guide.txt
├── templates/
│   └── index.html                  # Web UI (dark-theme single-page app)
└── chroma_db/                      # Vector store (auto-created, not committed)
```

---

## 🔧 Prerequisites

- **Python 3.9 or higher**
- **pip** (Python package installer)
- **IBM watsonx.ai account** with:
  - API Key
  - Project ID
  - Endpoint URL (e.g. `https://au-syd.ml.cloud.ibm.com`)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/anvirachavan-cloud/Recipe-generator.git
cd Recipe-generator
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the `sentence-transformers/all-MiniLM-L6-v2` embedding model (~90 MB) from Hugging Face.

---

## ⚙️ Configuration

Create a `config.env` file in the project root (copy from the template below):

```env
IBM_API_KEY=your_ibm_watsonx_api_key_here
IBM_PROJECT_ID=your_project_id_here
IBM_ENDPOINT_URL=https://au-syd.ml.cloud.ibm.com
FLASK_SECRET_KEY=your_random_secret_key
CHROMA_PERSIST_DIR=./chroma_db
DOCUMENTS_DIR=./documents
```

> ⚠️ **Never commit `config.env` to version control.** It is listed in `.gitignore`.

### IBM watsonx.ai Region Notes

| Region | Endpoint URL | Granite Model |
|--------|-------------|---------------|
| Sydney (au-syd) | `https://au-syd.ml.cloud.ibm.com` | `meta-llama/llama-3-3-70b-instruct` |
| US South | `https://us-south.ml.cloud.ibm.com` | `ibm/granite-3-8b-instruct` |
| EU (Frankfurt) | `https://eu-de.ml.cloud.ibm.com` | `ibm/granite-3-8b-instruct` |

To use IBM Granite natively, switch to `us-south` or `eu-de` and update `GRANITE_CHAT_MODEL_ID` in [`granite_llm.py`](granite_llm.py).

---

## ▶️ Running the Application

```bash
# Option 1 — Direct
python app.py

# Option 2 — Startup helper (installs deps + starts server)
python run.py

# Option 3 — Production with Gunicorn (Linux/macOS)
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

Then open your browser at **http://localhost:5000**

---

## 📡 API Reference

### `GET /api/health`
Check IBM watsonx.ai LLM connectivity.

**Response:**
```json
{
  "api": "ok",
  "llm": { "status": "healthy", "model": "meta-llama/llama-3-3-70b-instruct" }
}
```

---

### `POST /api/recipe/generate`
Generate a personalized recipe.

**Request Body:**
```json
{
  "query": "Make a quick pasta dinner",
  "diet_type": "vegetarian",
  "ingredients": ["pasta", "tomatoes", "garlic", "basil"],
  "allergies": ["nuts", "dairy"],
  "constraints": "30 minutes, one pot"
}
```

**Supported `diet_type` values:** `vegetarian`, `vegan`, `non-vegetarian`, `pescatarian`, `keto`, `diabetic-friendly`

**Response:**
```json
{
  "success": true,
  "recipe": "# Quick Vegetarian Pasta...",
  "warnings": [],
  "sources": [{ "source": "...", "category": "cookbook", "snippet": "..." }],
  "diet_type": "vegetarian",
  "allergen_alerts": []
}
```

---

### `POST /api/qa`
Ask any culinary question.

**Request Body:**
```json
{ "query": "What is the safe internal temperature for chicken?" }
```

**Response:**
```json
{
  "success": true,
  "answer": "The safe internal cooking temperature for chicken is 74°C / 165°F...",
  "sources": [...]
}
```

---

### `GET /api/agent/info`
Return agent metadata and vector store statistics.

---

### `POST /api/documents/reload`
Reload and re-index all documents from the `documents/` folder.

---

## 📚 Adding Your Own Documents

Place any of these supported file types in the `documents/` folder:

| Format | Extension | Examples |
|--------|-----------|---------|
| PDF | `.pdf` | Cookbooks, food science papers |
| Plain Text | `.txt` | Recipe collections, dietary guides |
| Word Document | `.docx` | Formatted recipe books |

Then either:
- Click **"Reload Knowledge Base"** in the About tab of the web UI, or
- Call `POST /api/documents/reload`

Document categories are automatically inferred from the filename:
- Files containing `cookbook`, `recipe`, `cooking` → `cookbook`
- Files containing `diet`, `nutrition`, `guideline`, `health` → `dietary_guideline`
- Files containing `allergy`, `allergen`, `safety`, `food_safety` → `food_safety`
- All others → `general_culinary`

---

## 🛡️ Guardrails & AI Ethics

The system implements five layers of safety:

```
Layer 1: Input Validation     → Blocks off-topic, harmful, or nonsensical queries
Layer 2: Dietary Compliance   → Detects meat/dairy in vegetarian/vegan recipes
Layer 3: Allergen Detection   → Scans output against all 14 major allergen groups
Layer 4: Food Safety Filters  → Flags unsafe cooking advice (undercooked meat, etc.)
Layer 5: Disclaimer Injection → Appends food safety disclaimer to every response
```

**Supported allergen groups:** Gluten, Dairy, Nuts/Peanuts, Eggs, Soy, Shellfish, Fish, Sesame, Sulfites, Celery, Mustard, Lupin, Molluscs.

---

## 🔧 Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **LLM** | IBM watsonx.ai (Granite / Llama 3.3) | — |
| **RAG Framework** | LangChain | ≥ 0.3 |
| **Vector Store** | ChromaDB | ≥ 0.5 |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | ≥ 3.0 |
| **Web Framework** | Flask | 3.0.3 |
| **Document Loaders** | LangChain Community | ≥ 0.3 |
| **Frontend** | Vanilla HTML / CSS / JavaScript | — |

---

## ⚠️ Disclaimer

> Recipes and information provided by this AI system are **for informational purposes only**. Always:
> - Cook meats to safe internal temperatures
> - Verify all ingredients for allergen content independently
> - Consult a qualified healthcare professional or registered dietitian for personalised dietary advice
>
> The AI may make errors. Always exercise personal judgment around food safety and consult professionals for medical dietary needs.

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
Made with ❤️ using <strong>IBM watsonx.ai</strong> · <strong>LangChain</strong> · <strong>ChromaDB</strong>
</div>
