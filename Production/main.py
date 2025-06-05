import os
import json
import time
import pickle
import asyncio
import functools
import httpx
from typing import List, Dict, Any
import numpy as np
import torch
import faiss
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertModel
from dotenv import load_dotenv
import anthropic
from openai import OpenAI
import logging
# ----------------------
# Load environment variables and initialize app
# ----------------------
load_dotenv()
app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ----------------------
# Basic Model and FAISS Initialization
# ----------------------
class UserRequest(BaseModel):
    user_id: str
    user_prompt: str


class AdjustedRecipeRequest(BaseModel):
    user_id: str
    food_title: str
    mealType: List[str]
    ingredients: str


def log_execution_time(func):
    """Decorator to log execution time for both sync and async functions."""
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            end = time.perf_counter()
            print(f"Execution time for {func.__name__}: {end - start:.4f} seconds")
            return result

        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"Execution time for {func.__name__}: {end - start:.4f} seconds")
            return result

        return sync_wrapper



API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not set in environment variables.")

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER")
OPENSEARCH_PWD = os.getenv("OPENSEARCH_PWD")


security = HTTPBearer()

claude_api_key = os.getenv("CLAUDE_API_KEY")
if not claude_api_key:
    raise EnvironmentError("Claude API key not set in environment variable CLAUDE_API_KEY")

claude_client = anthropic.Anthropic(api_key=claude_api_key)


def get_openai_client():
    api_key = os.getenv("PRODUCTION_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("PRODUCTION_OPENAI_API_KEY not found in environment variables")
    client = OpenAI(api_key=api_key)
    return client

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    print(f"Received API Key: {credentials.credentials}")
    print(f"Expected API Key: {API_KEY}")
    if credentials.scheme != "Bearer" or credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key",
        )


model_path = "Weights2/fine_tuned_distilbert_model"
tokenizer_path = "Weights2/fine_tuned_distilbert_tokenizer"
model = DistilBertModel.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
faiss_index_path = "description_faiss_ip_index.index"
metadata_path = "description_metadata.pkl"

index = None
try:
    index = faiss.read_index(faiss_index_path)
except Exception as e:
    print(f"Error loading FAISS index: {e}")

with open(metadata_path, 'rb') as f:
    description_metadata = pickle.load(f)
if index:
    index.nprobe = 50


# ----------------------
# Helper Functions
# ----------------------
@log_execution_time
async def get_user_profile(foodhak_user_id: str):
    """
    Asynchronously fetches the user profile using httpx.
    """
    url = OPENSEARCH_HOST
    query = {
        "query": {
            "match": {
                "foodhak_user_id": foodhak_user_id
            }
        }
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=query,
                                     auth=httpx.BasicAuth(OPENSEARCH_USER, OPENSEARCH_PWD))
    if response.status_code == 200:
        results = response.json()
        if results['hits']['total']['value'] > 0:
            result = results['hits']['hits'][0]['_source']
            user_health_goals = result.get("user_health_goals", [])
            primary_goal = next((goal for goal in user_health_goals if goal.get("user_goal", {}).get("is_primary")),
                                None)
            primary_goal_title = primary_goal["user_goal"].get("title") if primary_goal else user_health_goals[0][
                "user_goal"].get("title")
            nutrient_mapping = {
                "energy": "Energy (KCAL)",
                "protein": "Protein (G)",
                "fats": "Total Fat (G)",
                "saturated fat": "Saturated Fat (G)",
                "cholesterol": "Cholesterol (MG)",
                "sodium": "Sodium Na (MG)",
                "carbohydrates": "Total Carbohydrate (G)",
                "dietary fibre": "Dietary Fiber (G)",
                "vitamin c": "Vitamin C (MG)",
                "calcium": "Calcium (MG)",
                "iron": "Iron (MG)",
                "potassium": "Potassium K (MG)",
                "hydration": "Hydration (ML)"
            }
            nutrients_data = result.get("nutrients", {}).get("results", {})
            formatted_nutrients = {}
            for nutrient_type_list in nutrients_data.values():
                for nutrient_type in nutrient_type_list:
                    item_name = nutrient_type.get("nutrition_guideline", {}).get("item", "").strip()
                    item_name_lower = item_name.lower()
                    if item_name_lower in nutrient_mapping:
                        formatted_name = nutrient_mapping[item_name_lower]
                        formatted_nutrients[formatted_name] = str(nutrient_type.get("target_value"))
            profile_info = {
                "User Name": result.get("name"),
                "User Age": result.get("age"),
                "User Sex": result.get("sex"),
                "Primary Goal Title": primary_goal_title,
                "Goal Titles": [
                    goal_sub["title"] for goal in result.get("user_health_goals", [])
                    for key in ["user_goal", "user_goals"] if key in goal
                    for goal_sub in (goal[key] if isinstance(goal[key], list) else [goal[key]])
                ],
                "Ingredients to Recommend": [
                    {
                        "common_name": ingredient.get("common_name"),
                        "first_relationship_extract": ingredient["relationships"][0]["extracts"] if ingredient[
                            "relationships"] else None,
                        "first_relationship_url": ingredient["relationships"][0]["url"] if ingredient[
                            "relationships"] else None
                    }
                    for goal in result.get("user_health_goals", [])
                    for ingredient in goal.get("ingredients_to_recommend", [])
                ],
                "Ingredients to Avoid": [
                    {
                        "common_name": ingredient.get("common_name"),
                        "first_relationship_extract": ingredient["relationships"][0]["extracts"] if ingredient[
                            "relationships"] else None,
                        "first_relationship_url": ingredient["relationships"][0]["url"] if ingredient[
                            "relationships"] else None
                    }
                    for goal in result.get("user_health_goals", [])
                    for ingredient in goal.get("ingredients_to_avoid", [])
                ],
                "Dietary Restriction Name": result.get("dietary_restrictions", {}).get("name"),
                "Allergens Types": [allergen.get("type") for allergen in result.get("allergens", [])],
                "Daily Nutritional Requirement": formatted_nutrients
            }
            return profile_info
        else:
            print("No matching user profile found.")
            return None
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


@log_execution_time
def calculate_meal_macros(total_macros):
    return {
        "breakfast": {key: round(float(value) * 0.2, 2) for key, value in total_macros.items()},
        "lunch": {key: round(float(value) * 0.4, 2) for key, value in total_macros.items()},
        "dinner": {key: round(float(value) * 0.4, 2) for key, value in total_macros.items()},
    }


@log_execution_time
def extract_recipes(response_text):
    try:
        recipes = json.loads(response_text)
        recipes_with_titles = {}
        all_ingredients = set()

        for recipe in recipes:
            title = recipe.get("food_title") or recipe.get("title", "Untitled Recipe")
            meal_type = recipe.get("mealType", ["Unknown"])[0].capitalize()
            ingredients = recipe.get("ingredients", [])

            recipes_with_titles.setdefault(meal_type, []).append({
                "title": title,
                "ingredients": ingredients
            })

            if isinstance(ingredients, str):
                all_ingredients.update([i.strip() for i in ingredients.split(",") if i.strip()])
            elif isinstance(ingredients, list):
                all_ingredients.update(ingredients)

        # print("Extracted Recipes:", json.dumps(recipes_with_titles, indent=2))
        # print("Extracted Ingredients:", all_ingredients)

        return recipes_with_titles, list(all_ingredients)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from response: {e}")
        return {}, []


##################
def get_embedding(text: str) -> np.ndarray:
    """
    Get embeddings for a single text using the DistilBERT model.
    """
    try:
        # Tokenize the text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Get the model outputs
        with torch.no_grad():
            outputs = model(**inputs)

        # Use the [CLS] token embedding as the sentence embedding
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings.squeeze()

    except Exception as e:
        print(f"Error getting embedding for text '{text}': {e}")
        # Return a zero vector of the correct size as a fallback
        return np.zeros(model.config.hidden_size)


@log_execution_time
async def batch_process_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Process embeddings in batches to improve performance.
    """
    if not texts:
        return np.array([])

    results = []
    for i in range(0, len(texts), batch_size):
        try:
            batch = texts[i:i + batch_size]
            batch_embeddings = await asyncio.gather(
                *(async_get_embedding(text) for text in batch),
                return_exceptions=True
            )

            # Handle any exceptions in the batch
            processed_embeddings = []
            for j, embedding in enumerate(batch_embeddings):
                if isinstance(embedding, Exception):
                    print(f"Error processing text '{batch[j]}': {embedding}")
                    processed_embeddings.append(np.zeros(model.config.hidden_size))
                else:
                    processed_embeddings.append(embedding)

            results.extend(processed_embeddings)

        except Exception as e:
            print(f"Error processing batch {i // batch_size}: {e}")
            results.extend([np.zeros(model.config.hidden_size) for _ in batch])

    try:
        return np.stack(results)
    except Exception as e:
        print(f"Error stacking results: {e}")
        return np.array([])


async def async_get_embedding(text: str) -> np.ndarray:
    """
    Asynchronously get embeddings for a single text.
    """
    return await asyncio.to_thread(get_embedding, text)


@log_execution_time
async def batch_query_faiss(query_embeddings: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Query FAISS index with batched embeddings.
    """
    return await asyncio.to_thread(index.search, query_embeddings, top_k)


@log_execution_time
async def query_similar_descriptions_batch(query_texts: List[str], top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """
    Optimized version of query_similar_descriptions_batch using batch processing.
    """
    if not query_texts:
        return {}

    try:
        # Get embeddings for all texts in batches
        embeddings_np = await batch_process_embeddings(query_texts)

        if embeddings_np.size == 0:
            return {text: [] for text in query_texts}

        # Perform batch FAISS search
        try:
            D, I = await batch_query_faiss(embeddings_np, top_k)
        except Exception as e:
            print(f"Error in FAISS search: {e}")
            return {text: [] for text in query_texts}

        # Process results
        results = {}
        for i, text in enumerate(query_texts):
            query_results = []
            for rank in range(top_k):
                try:
                    idx_match = I[i, rank]
                    if 0 <= idx_match < len(description_metadata):
                        info = description_metadata[idx_match]
                        query_results.append({
                            "Description": info["Description"],
                            "Calories": info["Calories"],
                            "Protein": info["Protein"],
                            "Carbohydrate": info["Carbohydrate"],
                            "TotalFat": info["TotalFat"],
                            "Sodium": info["Sodium"],
                            "SaturatedFat": info["SaturatedFat"],
                            "Cholesterol": info["Cholesterol"],
                            "Sugar": info["Sugar"],
                            "Calcium": info["Calcium"],
                            "Iron": info["Iron"],
                            "Potassium": info["Potassium"],
                            "VitaminC": info["VitaminC"],
                            "VitaminE": info["VitaminE"],
                            "VitaminD": info["VitaminD"]
                        })
                except Exception as e:
                    print(f"Error processing result {rank} for text '{text}': {e}")
                    continue

            results[text] = query_results

        return results

    except Exception as e:
        print(f"Error in query_similar_descriptions_batch: {e}")
        return {text: [] for text in query_texts}


@log_execution_time
async def extract_ingredients_and_query_batch(ingredients_list: List[str], top_k: int = 1) -> Dict[
    str, List[Dict[str, Any]]]:
    """
    Optimized version of extract_ingredients_and_query_batch using batch processing.
    """
    if not ingredients_list:
        print("No ingredients to query.")
        return {}

    return await query_similar_descriptions_batch(ingredients_list, top_k)


#################


async def query_ingredients_in_chromadb_batch(
        ingredients_list: list,
        top_k: int = 5,
) -> Dict[str, List[str]]:
    """
    Queries an external vector store service to fetch similarity search results
    for a list of ingredients. Returns a dictionary mapping each ingredient
    to a list of retrieved page contents.
    """

    # API endpoint
    url = "https://ai-foodhak.com/chromadb_vecstore"
    headers = {"Content-Type": "application/json"}

    # Prepare query texts
    query_texts = [f"Health benefits of {ingredient} ?" for ingredient in ingredients_list]
    data = {"queries": query_texts, "count": top_k}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=data, headers=headers)
            response.raise_for_status()  # Raise exception for HTTP errors

            response_json = response.json()  # Parse JSON response

            # Process results and map them back to ingredients
            results = {}
            for ingredient, result in zip(ingredients_list, response_json):
                if isinstance(result, list):
                    results[ingredient] = [doc["page_content"] for doc in result]
                else:
                    results[ingredient] = []  # Empty list if no valid response

        except httpx.HTTPStatusError as http_err:
            print(f"HTTP error while querying vector store: {http_err}")
            results = {ingredient: [] for ingredient in ingredients_list}

        except Exception as e:
            print(f"Unexpected error: {e}")
            results = {ingredient: [] for ingredient in ingredients_list}

    return results


example_recipe_data = """
[   
    {
      "summary": "A balanced dinner bowl featuring lean protein and whole grains with colorful vegetables (Max: 50 chars)",
      "food_title": "Lean Beef and Brown Rice Bowl",
      "mealType": [
        "dinner"
      ],
      "servings": "1",
      "ready_in_minutes": "35",
      "ingredients": "4 oz lean beef strips, 3/4 cup brown rice, 4-5 olives, 1/2 cup bell peppers, 1/4 cup onions, 1 tbsp mixed herbs, 1/3 cup chickpeas, 1 tbsp lemon juice, 1/2 cup mixed vegetables",
      "Calories": "800",
      "Protein": "18.2",
      "Carbohydrates": "106.9",
      "Total Fat": "28.1",
      "Sodium": "380",
      "Saturated Fat": "5.9",
      "Cholesterol": "45",
      "Sugar": "4.8",
      "Calcium": "210",
      "Iron": "6.2",
      "Potassium": "890",
      "Vitamin C": "35",
      "Vitamin E": "3.8",
      "Vitamin D": "0.6",
      "cautions": [
        "Allergens here, if no allergens mention NONE"
      ],
      "diet_labels": [
        "Dairy-Free, Gluten-Free, Ketogenic, etc, if no diet_labels mention NONE"
      ],
      "instructions": "Recipe instructions... If recipe is going to be formatted as HTML, this can be a string field. Otherwise please return as a String array, e.g. [\"1. First step\", \"2. Second step\"]",
      "scientific_evidence_support": [
          {
            "description": "Describe how scientific evidence justifies the recommendation of these ingredients based on the user's profile. Additionally, explain the rationale behind suggesting this specific recipe for the user.",
            "source_url": "Pubmed URL"
          },
          {
            "description": "Describe how scientific evidence justifies the recommendation of these ingredients based on the user's profile. Additionally, explain the rationale behind suggesting this specific recipe for the user.",
            "source_url": "Pubmed URL"
          }
      ],
      "long_description": "long recipe description",
      "main_ingredients": ["Beef", "Rice", "Olives", "Bell Peppers"],
      "vitamin_mineral_claims": ["High in Vitamin C", "Source of Vitamin E", "High in Iron"]
    }
]
"""

example_recipe_title = """
[
  {
    "mealType": [
      "dinner"
    ],
    "food_title": "Recipe title here",
    "ingredients": "Reorder this list so that the main ingredients of this recipe appear first in the order; then include all other ingredients in their original order."
  },
  {
    "mealType": [
      "dinner"
    ],
    "food_title": "Recipe title here",
    "ingredients": "Reorder this list so that the main ingredients of this recipe appear first in the order; then include all other ingredients in their original order."
  }
]
"""


@log_execution_time
async def get_user_profile_and_macros(user_id: str) -> dict:
    """
    Helper function to get user profile and calculate meal macros.
    Returns a dictionary containing user profile, formatted macros, and raw macro data.
    """
    # Retrieve the user profile
    user_profile = await get_user_profile(user_id)
    if not user_profile:
        raise HTTPException(status_code=404, detail="User profile not found")

    # Calculate meal macros and format them
    user_macros = user_profile.get("Daily Nutritional Requirement", {})
    meal_macros = calculate_meal_macros(user_macros)
    meal_macros_formatted = ""
    for meal, values in meal_macros.items():
        meal_macros_formatted += f"{meal.title()}:\n"
        for nutrient, amount in values.items():
            meal_macros_formatted += f"  - {nutrient}: {amount}\n"

    return {
        "user_profile": user_profile,
        "meal_macros_formatted": meal_macros_formatted,
        "raw_macros": meal_macros
    }


# ===============================================
# Step 1: Reusable Function for Base Recipe Generation
# ===============================================
@log_execution_time
async def generate_base_recipe_internal(user_id: str, user_prompt: str) -> dict:
    # Get user profile and macros using the helper function
    profile_data = await get_user_profile_and_macros(user_id)
    user_profile = profile_data["user_profile"]
    meal_macros_formatted = profile_data["meal_macros_formatted"]

    # Prepare the base recipe prompt using the user profile and macros
    base_recipe_prompt = f"""
    You are a nutrition-aware recipe-generating assistant.
    Generate recipes based on the following user constraints:

    - **User Profile**: {user_profile}
    - **Target Macros (per meal)**: {meal_macros_formatted}
    - **Important Goals**:
    • Incorporate recommended ingredients **naturally** (avoid "forcing" any single ingredient).
    • Provide **variety** by not overusing the same ingredients repeatedly.
    • Align with any dietary restrictions or allergens in the user profile.
    • Keep the language friendly and approachable, ensuring clarity for non-expert home cooks.

    **Response Format**:
    {example_recipe_title}
    - The response must match the exact JSON structure above.
    - If multiple recipes are generated for the same meal, repeat the same JSON structure for each recipe.
    - Each recipe must be treated as a separate JSON snippet, nested in a JSON array.

    **Important Instructions**:
    - Do NOT ask follow-up questions.
    - Provide no additional text or explanation outside of the required JSON structure.
    - Maintain **professional but approachable** language.
    """
    try:
        system_instruction = f"""
        You are a nutrition-aware recipe-generating assistant. 
        Generate recipes based on user constraints from {user_profile} by analyzing recommended/avoided ingredients, allergens, dietary preferences, and goals.

        - **Do**:
        1. Use recommended ingredients **only** if they fit naturally.
        2. Ensure variety across multiple prompts/recipes (avoid repetitive meals).
        3. Adhere to macro targets from {meal_macros_formatted}.

        - **Do Not**:
        1. Force ingredients that do not complement the recipe.
        2. Provide additional commentary or disclaimers.
        3. Ask any follow-up questions.

        **Response Structure**:
        {example_recipe_title}
        - Repeat the JSON snippet for each separate recipe if multiple are requested.

        **Important**:
        - Return the output in valid JSON format only.
        - Maintain a consistent, user-friendly tone in your recipe descriptions.
        """
        try:
            with claude_client.messages.stream(
                model="claude-3-7-sonnet-20250219",
                max_tokens=3512,
                temperature=0,
                system=system_instruction,
                messages=[{"role": "user", "content": user_prompt + base_recipe_prompt}]
            ) as stream:
                texts = [text for text in stream.text_stream]
                base_recipe_output = ''.join(texts)
                cleaned_output = base_recipe_output.replace("```json", "").replace("```", "").strip()

        except Exception as e:
            error_message = str(e)
            logging.error(f"Error in Claude streaming: {e}")

            if "overloaded_error" in error_message or "529" in error_message:
                logging.warning("Claude is overloaded. Falling back to OpenAI...")

            try:
                openai_client = get_openai_client()
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt + base_recipe_prompt}
                    ],
                    stream=True
                )
                texts = []
                for chunk in response:
                    delta = chunk.choices[0].delta
                    texts.append(delta.get("content", ""))
                cleaned_output = ''.join(texts).replace("```json", "").replace("```", "").strip()
            except Exception as openai_exception:
                raise HTTPException(
                    status_code=500,
                    detail=f"Both Claude and OpenAI API calls failed. Claude error: {e}; OpenAI error: {openai_exception}"
                )
        try:
            # Parse the JSON to ensure it's valid and store it
            parsed_base_recipe = json.loads(cleaned_output)

            # Extract ingredients for future use
            ingredients_only = []
            for recipe in parsed_base_recipe:
                if isinstance(recipe.get("ingredients"), str):
                    ingredients_only.extend([i.strip() for i in recipe["ingredients"].split(",")])
                elif isinstance(recipe.get("ingredients"), list):
                    ingredients_only.extend(recipe["ingredients"])

            return {
                "base_recipe_output": cleaned_output,
                "parsed_recipe": parsed_base_recipe,
                "user_profile": user_profile,
                "meal_macros_formatted": meal_macros_formatted,
                "ingredients_only": ingredients_only
            }

        except json.JSONDecodeError as e:
            print(f"JSON Parsing Error: {e}")
            raise HTTPException(status_code=500, detail="Invalid recipe format generated")

    except Exception as e:
        print(f"Error generating base recipe: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate base recipe")


# ===============================================
# Endpoint 1: Generate Base Recipe
# ===============================================
@app.post("/generate_base_recipe/", dependencies=[Depends(verify_api_key)])
@log_execution_time
async def generate_base_recipe(request: UserRequest):
    data = await generate_base_recipe_internal(request.user_id, request.user_prompt)
    # Return the base recipe output as plain text
    return PlainTextResponse(content=data["base_recipe_output"], status_code=200)


# ===============================================
# Endpoint 2: Generate Adjusted Recipe (consumes base recipe stored from Endpoint 1)
# ===============================================

@app.post("/generate_adjusted_recipe/", dependencies=[Depends(verify_api_key)])
@log_execution_time
async def generate_adjusted_recipe(request: AdjustedRecipeRequest):
    # Get user profile and macros using the helper function
    profile_data = await get_user_profile_and_macros(request.user_id)
    user_profile = profile_data["user_profile"]
    meal_macros_formatted = profile_data["meal_macros_formatted"]

    # Create base recipe from request parameters
    base_recipe = [{
        "mealType": request.mealType,
        "food_title": request.food_title,
        "ingredients": request.ingredients
    }]
    # Query nutritional info if ingredients are present
    ingredients_only = []
    if isinstance(request.ingredients, str):
        ingredients_only.extend([i.strip() for i in request.ingredients.split(",")])
    elif isinstance(request.ingredients, list):
        ingredients_only.extend(request.ingredients)

    # Query nutritional info if ingredients are present
    if ingredients_only:
        nutritional_info, chromadb_nutritional_info = await asyncio.gather(
            extract_ingredients_and_query_batch(ingredients_only, top_k=1),
            query_ingredients_in_chromadb_batch(ingredients_only, top_k=1)
        )
        print("Nutritional Information USDA:", nutritional_info)
        print("Relhak Results:", chromadb_nutritional_info)
    else:
        nutritional_info, chromadb_nutritional_info = {}, {}

    # Build the adjusted recipe prompt using the exact base recipe
    adjusted_recipe_prompt = f"""
    You are a nutrition-aware recipe-generating assistant. Below is all the information you need to create customized recipes:

    1. **User Profile**:
    {user_profile}

    2. **Target Macros for Breakfast, Lunch, and Dinner**:
    {meal_macros_formatted}

    3. **Base Recipe(s) to Work From**:
    {json.dumps(base_recipe, indent=2)}

    4. **Nutritional Information (per 100g) from FAISS**:
    {nutritional_info}
    - If any ingredients in {nutritional_info} are missing from {ingredients_only}, 
        generate nutritional info for those missing ingredients using your own knowledge.

    5. **Scientific Studies for Each Ingredient ({ingredients_only}) from PubMed**:
    {chromadb_nutritional_info}

    ---

    ### Your Tasks

    #### A. Detailed Recipe Instructions
    - For each recipe in the base recipe, provide step-by-step cooking instructions.
    - Use **exact measurements** (cups, tablespoons, etc.) where possible.
    - Include short, user-friendly clarifications (e.g., "simmer until fragrant," "chop finely").

    #### B. Accurate Nutritional Breakdown
    - Provide key nutrients for each recipe:
    - **Calories, Protein, Carbohydrate, TotalFat, Sodium, SaturatedFat, Cholesterol, Sugar, Calcium, Iron, Potassium, VitaminC, VitaminE, VitaminD**
    - Ensure these values align with the target macros in {meal_macros_formatted}.

    #### C. Macro Compliance
    - Confirm total Calories, Protein, Carbs, and Fats match the user's target macros.
    - State approximate final totals.

    #### D. Scientific Justification and Nutritional Claims
    - Summarize how the scientific evidence from {chromadb_nutritional_info} supports using these ingredients in a **user-friendly** manner.
    - If the recipe meets any **"source of"** or **"high in"** thresholds (based on your Vitamin & Mineral Claims table), explicitly mention these claims. 
    - For example: "This recipe can be labeled as a **Source of Vitamin C** since it contains more than 12 mg per serving. Research shows that adequate Vitamin C intake supports immune function (PubMed ID XXXX)."
    - Use **concise, everyday language** (avoid heavy scientific jargon).
    - Focus on practical benefits (e.g., "rich in antioxidants," "supports heart health") and connect them to the labeling claims (e.g., "High in Iron").

    #### E. JSON Subsection
    - For each recipe, include a JSON snippet in the exact format: {example_recipe_data}
    - Populate all relevant data (ingredients, steps, nutritional data, etc.).
    - Even for multiple recipes treat each one as a separate recipe with JSON snippet for **each** recipes nested in JSON object array.


    #### F. General Instructions
    - Do NOT ask any follow-up questions.
    - No extra text or explanations beyond the required format.
    - Return the response in **valid JSON**.
    - Use the recipe name(s) exactly as given in the base recipes.
    - Do NOT generate recipes beyond those provided in the base recipe.
    """
    print("Adjusted recipe prompt is:", adjusted_recipe_prompt)
    try:
        system_instruction = f"""
        You are a nutrition-aware recipe-generating assistant.
        Below is the 'Vitamin & Mineral Claims' table you must reference to determine if each recipe should be labeled as a 'source of' or 'high in' a particular nutrient. 
        Use the total amounts per recipe to decide whether the recipe qualifies:

        | **Vitamin/Mineral**       | **"Source of" Threshold** | **"High in" Threshold** |
        |---------------------------|--------------------------|------------------------|
        | **Vitamin A (μg)**        | 120                      | 240                    |
        | **Vitamin D (μg)**        | 0.75                     | 1.5                    |
        | **Vitamin E (mg)**        | 1.8                      | 3.6                    |
        | **Vitamin K (μg)**        | 11.25                    | 22.5                   |
        | **Vitamin C (mg)**        | 12                       | 24                     |
        | **Thiamin (mg)**          | 0.165                    | 0.33                   |
        | **Riboflavin B2 (mg)**    | 0.21                     | 0.42                   |
        | **Niacin (mg)**           | 2.4                      | 4.8                    |
        | **Vitamin B6 (mg)**       | 0.21                     | 0.42                   |
        | **Folic Acid (μg)**       | 30                       | 60                     |
        | **Vitamin B12 (μg)**      | 0.375                    | 0.75                   |
        | **Biotin B7 (μg)**        | 7.5                      | 15                     |
        | **Pantothenic B5 (mg)**   | 0.9                      | 1.8                    |
        | **Potassium (mg)**        | 300                      | 600                    |
        | **Chloride (mg)**         | 120                      | 240                    |
        | **Calcium (mg)**          | 120                      | 240                    |
        | **Phosphorus (mg)**       | 105                      | 210                    |
        | **Magnesium (mg)**        | 56.25                    | 112.5                  |
        | **Iron (mg)**             | 2.1                      | 4.2                    |
        | **Zinc (mg)**             | 1.5                      | 3                      |
        | **Copper (mg)**           | 0.15                     | 0.3                    |
        | **Manganese (mg)**        | 0.3                      | 0.6                    |
        | **Fluoride (mg)**         | 0.525                    | 1.05                   |
        | **Selenium (μg)**         | 8.25                     | 16.5                   |
        | **Chromium (μg)**         | 6                        | 12                     |
        | **Molybdenum (μg)**       | 7.5                      | 15                     |
        | **Iodine (μg)**           | 22.5                     | 45                     |

        **Instructions**:
        1. For each recipe, calculate its total nutrient levels to decide if it meets or exceeds "Source of" or "High in" thresholds.
        2. Label any qualifying recipe accordingly (e.g., "Source of Vitamin C", "High in Calcium") and **include** that label in the recipe output and in the "Scientific Justification" explanation.
        3. Always provide the following nutrients in the breakdown:
        - **Calories, Protein, Carbohydrate, TotalFat, Sodium, SaturatedFat, Cholesterol, Sugar, Calcium, Iron, Potassium, VitaminC, VitaminE, VitaminD**.
        4. Ensure the recipe aligns with the user's target macros: {meal_macros_formatted}.
        5. Return each recipe in valid JSON array, using the **exact** recipe name from the base recipe.
        6. Do not provide any extra commentary or ask follow-up questions.
        """
        try:
            with claude_client.messages.stream(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=8192,
                    temperature=0,
                    system=system_instruction,
                    messages=[{"role": "user", "content": adjusted_recipe_prompt}]
            ) as stream:
                texts = [text for text in stream.text_stream]
                final_adjusted_recipe = ''.join(texts)
                final_adjusted_recipe = final_adjusted_recipe.replace("```json", "").replace("```", "").strip()
            print("Final Adjusted Recipe Output from Claude:")
            print(final_adjusted_recipe)
        except Exception as e:
            error_message = str(e)
            logging.error(f"Error in Claude streaming: {e}")

            if "overloaded_error" in error_message or "529" in error_message:
                logging.warning("Claude is overloaded. Falling back to OpenAI...")

                try:
                    openai_client = get_openai_client()
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": adjusted_recipe_prompt}
                        ],
                        stream=True
                    )
                    texts = []
                    for chunk in response:
                        delta = chunk.choices[0].delta
                        texts.append(delta.get("content", ""))
                    final_adjusted_recipe  = ''.join(texts).replace("```json", "").replace("```", "").strip()
                except Exception as openai_exception:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Both Claude and OpenAI API calls failed. Claude error: {e}; OpenAI error: {openai_exception}"
                    )
    except Exception as e:
        print(f"Error generating adjusted recipe: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate adjusted recipe")
    return PlainTextResponse(content=final_adjusted_recipe, status_code=200)


@app.get("/health")
async def health_check():
    print("Production: Recipe Generator is up and running")
    return {"status": "healthy", "message": "Recipe Generator Service is up and running."}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
