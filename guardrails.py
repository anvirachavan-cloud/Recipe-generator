"""
Guardrails Module
Applies AI ethics, safety filters, and content validation for the Recipe RAG Agent.
"""

import re
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

# ── Unsafe content patterns ──────────────────────────────────────────────────
UNSAFE_COOKING_PATTERNS = [
    r"\braw\s+(chicken|pork|poultry|ground\s+beef)\b",
    r"\bundercook\b",
    r"\beat\s+raw\s+eggs?\b",
    r"\bdo\s+not\s+wash\s+hands\b",
    r"\bignore\s+expir",
    r"\bpoison(ous|ing)?\b.*\bmushroom",
    r"\bwild\s+berry\b",
    r"\bserve\s+immediately\s+without\s+cooking\b",
]

UNSAFE_INGREDIENT_PATTERNS = [
    r"\bbleach\b",
    r"\bchemical\b.*\bingredient\b",
    r"\balcohol\b.*\bchild\b",
    r"\bpoisonous\b",
    r"\btoxic\b.*\bingredient\b",
]

# ── Allergy-triggering ingredients map ──────────────────────────────────────
COMMON_ALLERGENS = {
    "gluten": ["wheat", "barley", "rye", "flour", "bread", "pasta", "semolina", "spelt"],
    "dairy": ["milk", "butter", "cheese", "cream", "yogurt", "whey", "lactose", "ghee"],
    "nuts": ["peanut", "almond", "walnut", "cashew", "pistachio", "hazelnut", "pecan", "macadamia"],
    "eggs": ["egg", "eggs", "mayonnaise", "meringue"],
    "soy": ["soy", "soya", "tofu", "tempeh", "miso", "edamame"],
    "shellfish": ["shrimp", "crab", "lobster", "prawn", "crayfish", "scallop", "clam"],
    "fish": ["salmon", "tuna", "cod", "tilapia", "anchovy", "sardine", "mackerel"],
    "sesame": ["sesame", "tahini"],
    "sulfites": ["wine", "dried fruit", "pickled", "vinegar"],
}

# ── Dietary compliance checks ────────────────────────────────────────────────
NON_VEG_INGREDIENTS = [
    "chicken", "beef", "pork", "lamb", "mutton", "fish", "shrimp", "prawn",
    "crab", "lobster", "salmon", "tuna", "turkey", "duck", "venison", "bacon",
    "ham", "sausage", "pepperoni", "anchovies", "gelatin", "lard",
]

NON_VEGAN_INGREDIENTS = NON_VEG_INGREDIENTS + [
    "milk", "butter", "cheese", "cream", "yogurt", "egg", "eggs",
    "honey", "whey", "ghee", "mayonnaise",
]

# ── Forbidden query topics ────────────────────────────────────────────────────
OFF_TOPIC_PATTERNS = [
    r"\b(hack|exploit|bypass|crack)\b",
    r"\b(weapon|bomb|explosive)\b",
    r"\b(illegal|narcotic|drug)\b",
    r"\bsuicid\b",
    r"\bharm\s+(myself|yourself|someone)\b",
]

FOOD_SAFETY_DISCLAIMER = """
---
⚠️ **FOOD SAFETY & ALLERGY DISCLAIMER**

- Always wash hands thoroughly before and after handling food.
- Cook meats to safe internal temperatures (chicken: 165°F/74°C; beef: 145°F/63°C; pork: 145°F/63°C).
- People with food allergies should carefully verify ALL ingredients before consuming any recipe.
- If you have severe allergies, consult a healthcare professional before trying new recipes.
- The recipes and advice provided are for informational purposes only and should not replace professional dietary or medical guidance.
- Always check product labels for hidden allergens and cross-contamination warnings.
- When in doubt about ingredient safety or freshness, discard and use fresh alternatives.

*This AI system applies safety guardrails, but always exercise personal judgment and consult professionals for dietary needs.*
---
"""


class GuardrailsEngine:
    """
    Applies multi-layer safety and ethics guardrails to inputs and outputs
    of the Recipe RAG Agent.
    """

    def __init__(self):
        self.unsafe_cooking = [re.compile(p, re.IGNORECASE) for p in UNSAFE_COOKING_PATTERNS]
        self.unsafe_ingredients = [re.compile(p, re.IGNORECASE) for p in UNSAFE_INGREDIENT_PATTERNS]
        self.off_topic = [re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS]

    # ── Input validation ─────────────────────────────────────────────────────

    def validate_query(self, query: str) -> Tuple[bool, str]:
        """
        Validate user query before processing.
        Returns (is_safe, reason_if_unsafe).
        """
        # Off-topic / harmful intent check
        for pattern in self.off_topic:
            if pattern.search(query):
                return False, "Your query appears to contain content outside the scope of culinary assistance. Please ask about recipes, ingredients, or cooking techniques."

        # Minimum meaningful query length
        if len(query.strip()) < 5:
            return False, "Please provide a more detailed question about recipes or cooking."

        return True, ""

    def validate_user_preferences(
        self,
        diet_type: str,
        ingredients: List[str],
        allergies: List[str],
    ) -> Tuple[bool, str]:
        """Validate user-supplied preferences for internal consistency."""
        valid_diets = {"vegetarian", "vegan", "non-vegetarian", "veg", "non-veg", "pescatarian", "keto", "diabetic-friendly"}
        if diet_type.lower() not in valid_diets:
            return False, f"Unrecognised diet type '{diet_type}'. Supported: {', '.join(sorted(valid_diets))}"
        return True, ""

    # ── Output validation ────────────────────────────────────────────────────

    def validate_response(self, response: str) -> Tuple[bool, str, str]:
        """
        Validate LLM-generated response before returning to the user.
        Returns (is_safe, warning_message, cleaned_response).
        """
        warnings = []

        for pattern in self.unsafe_cooking:
            if pattern.search(response):
                warnings.append("⚠️ Unsafe cooking practice detected in response — please follow safe food handling guidelines.")
                break

        for pattern in self.unsafe_ingredients:
            if pattern.search(response):
                warnings.append("⚠️ A potentially unsafe ingredient reference was detected. Always verify ingredient safety.")
                break

        warning_text = "\n".join(warnings)
        return True, warning_text, response  # We warn but don't block valid recipes

    # ── Dietary compliance ────────────────────────────────────────────────────

    def check_dietary_compliance(
        self,
        recipe_text: str,
        diet_type: str,
    ) -> Tuple[bool, List[str]]:
        """
        Check if a generated recipe is compliant with the requested diet type.
        Returns (is_compliant, list_of_violations).
        """
        violations = []
        recipe_lower = recipe_text.lower()
        diet = diet_type.lower()

        if diet in ("vegetarian", "veg"):
            for ingredient in NON_VEG_INGREDIENTS:
                if re.search(rf"\b{ingredient}\b", recipe_lower):
                    violations.append(ingredient)

        elif diet in ("vegan",):
            for ingredient in NON_VEGAN_INGREDIENTS:
                if re.search(rf"\b{ingredient}\b", recipe_lower):
                    violations.append(ingredient)

        return len(violations) == 0, violations

    # ── Allergy detection ─────────────────────────────────────────────────────

    def detect_allergens(
        self,
        recipe_text: str,
        user_allergies: List[str],
    ) -> List[str]:
        """
        Detect allergens in recipe text that match the user's declared allergies.
        Returns list of detected allergen categories.
        """
        triggered = []
        recipe_lower = recipe_text.lower()

        for allergy in user_allergies:
            allergy_key = allergy.lower().strip()
            # Direct match
            if re.search(rf"\b{re.escape(allergy_key)}\b", recipe_lower):
                triggered.append(allergy_key)
                continue
            # Mapped ingredient match
            if allergy_key in COMMON_ALLERGENS:
                for ingredient in COMMON_ALLERGENS[allergy_key]:
                    if re.search(rf"\b{re.escape(ingredient)}\b", recipe_lower):
                        triggered.append(f"{allergy_key} ({ingredient})")
                        break

        return triggered

    # ── Disclaimer injection ──────────────────────────────────────────────────

    def inject_disclaimer(self, response: str) -> str:
        """Append the food safety and allergy disclaimer to any response."""
        return response.strip() + "\n\n" + FOOD_SAFETY_DISCLAIMER

    def build_allergy_warning(self, triggered_allergens: List[str]) -> str:
        """Build a formatted allergy warning block."""
        if not triggered_allergens:
            return ""
        allergen_list = ", ".join(triggered_allergens)
        return (
            f"\n\n🚨 **ALLERGY ALERT**: This recipe may contain or reference the following "
            f"allergens relevant to your restrictions: **{allergen_list}**. "
            f"Please review the ingredient list carefully and consider safe substitutions."
        )

    def build_diet_violation_warning(self, violations: List[str]) -> str:
        """Build a formatted dietary violation warning."""
        if not violations:
            return ""
        items = ", ".join(violations)
        return (
            f"\n\n⚠️ **DIETARY NOTE**: The following ingredients may not align with your "
            f"selected diet preference: **{items}**. "
            f"Consider substituting or omitting these ingredients."
        )


# Singleton instance
guardrails = GuardrailsEngine()
