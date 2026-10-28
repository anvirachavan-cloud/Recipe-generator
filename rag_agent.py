"""
Recipe RAG Agent
Core agentic AI module that orchestrates retrieval, generation, and guardrails
using IBM Granite LLM and ChromaDB vector store.
"""

import logging
from typing import Dict, Any, List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from granite_llm import get_recipe_llm, get_qa_llm
from document_ingestion import get_or_create_vector_store, get_retriever
from guardrails import guardrails, FOOD_SAFETY_DISCLAIMER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Prompt Templates ──────────────────────────────────────────────────────────

RECIPE_GENERATION_PROMPT = PromptTemplate(
    input_variables=["context", "question", "diet_type", "ingredients", "allergies", "constraints"],
    template="""You are a professional chef and culinary assistant powered by IBM Granite AI. 
Your role is to generate personalized, safe, and delicious recipes based on the user's preferences and available culinary knowledge.

### Culinary Knowledge Base Context:
{context}

### User Preferences:
- Diet Type: {diet_type}
- Available Ingredients: {ingredients}
- Allergy Restrictions: {allergies}
- Cooking Constraints: {constraints}

### User Request:
{question}

### Instructions:
1. Generate a complete, personalized recipe that strictly respects the diet type and allergy restrictions.
2. Use only the available ingredients listed, supplementing with common pantry staples if necessary.
3. Follow safe food handling and cooking temperature guidelines.
4. Structure the recipe clearly with: Recipe Name, Description, Servings, Prep/Cook Time, Ingredients List, Step-by-Step Instructions, Nutritional Notes, and Chef Tips.
5. Be creative, helpful, and encouraging in tone.
6. If the context contains relevant cookbook or dietary information, incorporate those insights.
7. Never suggest ingredients that conflict with stated allergies or diet type.

### Generated Recipe:
""",
)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a knowledgeable culinary AI assistant powered by IBM Granite. 
You answer questions about cooking, recipes, nutrition, food safety, and dietary guidelines accurately and helpfully.

### Knowledge Base Context:
{context}

### User Question:
{question}

### Instructions:
1. Answer based on the provided context and your culinary knowledge.
2. Be accurate, concise, and helpful.
3. If the question relates to food safety, always err on the side of caution.
4. If you don't know the answer, say so clearly rather than guessing.
5. Use a warm, encouraging, and educational tone.

### Answer:
""",
)

AGENT_SYSTEM_PROMPT = """You are Chef Granite, an expert culinary AI assistant powered by IBM Granite LLM.
You help users discover recipes, understand cooking techniques, learn about nutrition, and navigate dietary restrictions.
You are creative, knowledgeable, safety-conscious, and always respect user preferences.
You ground all responses in evidence from culinary documents and food science.
"""


class RecipeRAGAgent:
    """
    Agentic RAG system for recipe generation and culinary Q&A.
    Orchestrates:
    - User preference collection
    - Vector retrieval from culinary knowledge base
    - IBM Granite LLM-based generation
    - Guardrails and safety validation
    - Personalized response delivery
    """

    def __init__(self, force_rebuild: bool = False):
        logger.info("Initializing Recipe RAG Agent...")
        self.vectorstore: Chroma = get_or_create_vector_store(force_rebuild=force_rebuild)
        self.retriever = get_retriever(self.vectorstore, k=5)
        self.recipe_llm = get_recipe_llm()
        self.qa_llm = get_qa_llm()
        logger.info("Recipe RAG Agent initialized successfully.")

    def _retrieve_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant document chunks for a query."""
        docs: List[Document] = self.retriever.invoke(query)
        context_parts = []
        for i, doc in enumerate(docs[:k], 1):
            category = doc.metadata.get("category", "general")
            source = doc.metadata.get("source", "knowledge base")
            context_parts.append(
                f"[Source {i} — {category}]:\n{doc.page_content.strip()}"
            )
        return "\n\n".join(context_parts) if context_parts else "No specific documents found. Using general culinary knowledge."

    def generate_recipe(
        self,
        query: str,
        diet_type: str = "non-vegetarian",
        ingredients: List[str] = None,
        allergies: List[str] = None,
        constraints: str = "None",
    ) -> Dict[str, Any]:
        """
        Generate a personalized recipe using RAG + IBM Granite.
        
        Args:
            query: User's recipe request.
            diet_type: User's diet preference.
            ingredients: Available ingredients.
            allergies: User's allergy restrictions.
            constraints: Cooking constraints (time, equipment, etc.).
        
        Returns:
            Dict with recipe, warnings, sources, and metadata.
        """
        ingredients = ingredients or []
        allergies = allergies or []

        # ── Guardrail: validate query ────────────────────────────────────────
        is_safe, reason = guardrails.validate_query(query)
        if not is_safe:
            return {
                "success": False,
                "error": reason,
                "recipe": "",
                "warnings": [],
                "sources": [],
            }

        # ── Guardrail: validate preferences ─────────────────────────────────
        pref_valid, pref_reason = guardrails.validate_user_preferences(
            diet_type, ingredients, allergies
        )
        if not pref_valid:
            return {
                "success": False,
                "error": pref_reason,
                "recipe": "",
                "warnings": [],
                "sources": [],
            }

        # ── Retrieve culinary context ────────────────────────────────────────
        enriched_query = f"{diet_type} recipe: {query}. Ingredients: {', '.join(ingredients) if ingredients else 'flexible'}. Allergies: {', '.join(allergies) if allergies else 'none'}."
        context = self._retrieve_context(enriched_query)

        # ── Build enriched prompt ────────────────────────────────────────────
        ingredients_str = ", ".join(ingredients) if ingredients else "flexible — use common pantry items"
        allergies_str = ", ".join(allergies) if allergies else "None declared"

        full_prompt = RECIPE_GENERATION_PROMPT.format(
            context=context,
            question=query,
            diet_type=diet_type,
            ingredients=ingredients_str,
            allergies=allergies_str,
            constraints=constraints,
        )

        # ── Generate with IBM Granite ────────────────────────────────────────
        logger.info("Generating recipe with IBM Granite for query: %s", query[:80])
        try:
            raw_response = self.recipe_llm.invoke(full_prompt)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return {
                "success": False,
                "error": f"Recipe generation failed: {str(exc)}",
                "recipe": "",
                "warnings": [],
                "sources": [],
            }

        # ── Output guardrails ────────────────────────────────────────────────
        is_safe, warning, cleaned = guardrails.validate_response(raw_response)
        warnings = [warning] if warning else []

        # Dietary compliance check
        compliant, violations = guardrails.check_dietary_compliance(cleaned, diet_type)
        if not compliant:
            diet_warn = guardrails.build_diet_violation_warning(violations)
            if diet_warn:
                warnings.append(diet_warn)

        # Allergy detection
        triggered = guardrails.detect_allergens(cleaned, allergies)
        if triggered:
            allergy_warn = guardrails.build_allergy_warning(triggered)
            if allergy_warn:
                warnings.append(allergy_warn)

        # Append all warnings + disclaimer to recipe
        warning_block = "\n".join(warnings) if warnings else ""
        final_recipe = cleaned + (f"\n{warning_block}" if warning_block else "")
        final_recipe = guardrails.inject_disclaimer(final_recipe)

        # ── Collect source documents ─────────────────────────────────────────
        source_docs = self.retriever.invoke(enriched_query)
        sources = [
            {
                "source": d.metadata.get("source", "Knowledge Base"),
                "category": d.metadata.get("category", "general"),
                "snippet": d.page_content[:200] + "...",
            }
            for d in source_docs[:3]
        ]

        return {
            "success": True,
            "recipe": final_recipe,
            "warnings": warnings,
            "sources": sources,
            "diet_type": diet_type,
            "allergen_alerts": triggered,
        }

    def answer_query(self, query: str, context_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Answer a general culinary Q&A query using RAG.
        
        Args:
            query: User's culinary question.
            context_filter: Optional category filter for retrieval.
        
        Returns:
            Dict with answer, sources, and metadata.
        """
        # ── Guardrail: validate query ────────────────────────────────────────
        is_safe, reason = guardrails.validate_query(query)
        if not is_safe:
            return {
                "success": False,
                "error": reason,
                "answer": "",
                "sources": [],
            }

        # ── Retrieve context ─────────────────────────────────────────────────
        context = self._retrieve_context(query)

        # ── Build prompt and generate ────────────────────────────────────────
        full_prompt = QA_PROMPT.format(context=context, question=query)
        logger.info("Answering Q&A with IBM Granite: %s", query[:80])

        try:
            raw_response = self.qa_llm.invoke(full_prompt)
        except Exception as exc:
            logger.error("Q&A generation failed: %s", exc)
            return {
                "success": False,
                "error": f"Query answering failed: {str(exc)}",
                "answer": "",
                "sources": [],
            }

        # ── Output guardrails ────────────────────────────────────────────────
        _, warning, cleaned = guardrails.validate_response(raw_response)
        final_answer = cleaned
        if warning:
            final_answer += f"\n\n{warning}"
        final_answer = guardrails.inject_disclaimer(final_answer)

        source_docs = self.retriever.invoke(query)
        sources = [
            {
                "source": d.metadata.get("source", "Knowledge Base"),
                "category": d.metadata.get("category", "general"),
                "snippet": d.page_content[:200] + "...",
            }
            for d in source_docs[:3]
        ]

        return {
            "success": True,
            "answer": final_answer,
            "sources": sources,
        }

    def get_agent_info(self) -> Dict[str, Any]:
        """Return agent metadata and health information."""
        try:
            doc_count = self.vectorstore._collection.count()
        except Exception:
            doc_count = 0
        return {
            "agent_name": "Chef Granite — Recipe RAG Agent",
            "model": "Meta Llama 3.3 70B Instruct (IBM watsonx.ai au-syd)",
            "vector_store": "ChromaDB",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "document_chunks": doc_count,
            "capabilities": [
                "Personalized Recipe Generation",
                "Culinary Q&A",
                "Dietary Restriction Compliance",
                "Allergy Detection",
                "Food Safety Guardrails",
            ],
        }

    def reload_documents(self) -> Dict[str, Any]:
        """Reload and re-index all documents."""
        logger.info("Reloading documents and rebuilding vector store...")
        self.vectorstore = get_or_create_vector_store(force_rebuild=True)
        self.retriever = get_retriever(self.vectorstore, k=5)
        count = self.vectorstore._collection.count()
        return {"success": True, "message": f"Vector store rebuilt with {count} chunks."}


# ── Module-level singleton (lazy init) ───────────────────────────────────────
_agent_instance: Optional[RecipeRAGAgent] = None


def get_agent() -> RecipeRAGAgent:
    """Get or create the global RecipeRAGAgent singleton."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = RecipeRAGAgent()
    return _agent_instance
