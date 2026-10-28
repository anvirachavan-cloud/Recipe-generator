"""
Flask Web Application — Recipe Generator RAG Agent
Provides REST API endpoints and serves the web UI.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load config.env for environment variables
load_dotenv("config.env")

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "recipe-rag-secret")
CORS(app)

# ── Lazy agent initialization ─────────────────────────────────────────────────
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        from rag_agent import RecipeRAGAgent
        logger.info("Initializing RAG Agent (first request)...")
        _agent = RecipeRAGAgent()
    return _agent


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main web UI."""
    template_path = Path("templates/index.html")
    if template_path.exists():
        return send_from_directory("templates", "index.html")
    return "<h1>Recipe Generator RAG Agent</h1><p>UI not found.</p>", 404


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        from granite_llm import health_check
        status = health_check()
        return jsonify({"api": "ok", "llm": status}), 200
    except Exception as exc:
        return jsonify({"api": "ok", "llm": {"status": "unavailable", "error": str(exc)}}), 200


@app.route("/api/agent/info", methods=["GET"])
def agent_info():
    """Return agent metadata."""
    try:
        info = get_agent().get_agent_info()
        return jsonify(info), 200
    except Exception as exc:
        logger.error("Agent info error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/recipe/generate", methods=["POST"])
def generate_recipe():
    """
    Generate a personalized recipe.
    
    Expected JSON body:
    {
        "query": "Make a pasta dish",
        "diet_type": "vegetarian",
        "ingredients": ["pasta", "tomatoes", "garlic"],
        "allergies": ["nuts", "dairy"],
        "constraints": "30 minutes, no oven"
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"success": False, "error": "Query is required."}), 400

    diet_type = data.get("diet_type", "non-vegetarian")
    ingredients_raw = data.get("ingredients", [])
    # Accept either a list or a comma-separated string
    if isinstance(ingredients_raw, str):
        ingredients = [i.strip() for i in ingredients_raw.split(",") if i.strip()]
    else:
        ingredients = [str(i).strip() for i in ingredients_raw if str(i).strip()]

    allergies_raw = data.get("allergies", [])
    if isinstance(allergies_raw, str):
        allergies = [a.strip() for a in allergies_raw.split(",") if a.strip()]
    else:
        allergies = [str(a).strip() for a in allergies_raw if str(a).strip()]

    constraints = data.get("constraints", "None")

    logger.info(
        "Recipe request — diet: %s, ingredients: %s, allergies: %s",
        diet_type, ingredients[:5], allergies,
    )

    try:
        result = get_agent().generate_recipe(
            query=query,
            diet_type=diet_type,
            ingredients=ingredients,
            allergies=allergies,
            constraints=constraints,
        )
        return jsonify(result), 200 if result["success"] else 400
    except Exception as exc:
        logger.error("Recipe generation error: %s", exc)
        return jsonify({"success": False, "error": f"Internal error: {str(exc)}"}), 500


@app.route("/api/qa", methods=["POST"])
def culinary_qa():
    """
    Answer a culinary Q&A question.
    
    Expected JSON body:
    {
        "query": "What is the difference between baking soda and baking powder?"
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"success": False, "error": "Query is required."}), 400

    logger.info("Q&A request: %s", query[:100])
    try:
        result = get_agent().answer_query(query)
        return jsonify(result), 200 if result["success"] else 400
    except Exception as exc:
        logger.error("Q&A error: %s", exc)
        return jsonify({"success": False, "error": f"Internal error: {str(exc)}"}), 500


@app.route("/api/documents/reload", methods=["POST"])
def reload_documents():
    """Reload and re-index all documents in the documents directory."""
    try:
        result = get_agent().reload_documents()
        return jsonify(result), 200
    except Exception as exc:
        logger.error("Document reload error: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting Recipe Generator RAG Agent on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)
