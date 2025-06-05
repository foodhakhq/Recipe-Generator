````markdown
# 🥗 Recipe Generator AI Service

**Intelligent, nutrition-aware recipe creation and customization for Foodhak users.  
Personalized to user profile, macros, goals, allergies, and science-backed ingredient claims.  
LLM: Anthropic Claude (fallback: OpenAI GPT-4o).**

---

## 🏗️ What does it do?

- Generates **personalized recipes** (breakfast/lunch/dinner/snack) based on:
    - User’s health goals, macro targets, allergies, dietary restrictions, and ingredient preferences.
    - Evidence from nutrition databases and PubMed.
- Returns **clean, valid JSON**: easy for apps or UIs.
- Produces detailed nutrition breakdown and science-backed justification for every recipe.
- Supports **base recipe generation** and subsequent *adjustment* for macro or ingredient tweaks.

---

## 🚀 Endpoints

| Method | Endpoint                    | Description                       |
|--------|----------------------------|-----------------------------------|
| POST   | `/generate_base_recipe/`   | Generate base recipe for a user   |
| POST   | `/generate_adjusted_recipe/`| Adjust an existing recipe         |
| GET    | `/health`                  | Health check                      |

**All POST endpoints require:**  
`Authorization: Bearer <API_KEY>`

---

## 🧩 How it works

1. **Base Recipe Generation**
    - Input: user ID + freeform meal/recipe prompt.
    - Returns: 1–N candidate recipes, with ingredients prioritized by user profile.

2. **Recipe Adjustment**
    - Input: recipe title, meal type(s), and full ingredient list (from base recipe), plus user ID.
    - Returns: new recipe version with updated macros, step-by-step instructions, and full nutrition breakdown.
    - *Science/claims* block included for consumer trust and labeling.

---

## 🛠️ Example Usage

### 1. Generate Base Recipe

```bash
curl -X POST https://ai-foodhak.com/generate_base_recipe/ \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "123456", "user_prompt": "I want a protein-rich vegetarian lunch bowl with no nuts."}'
````

#### Example JSON Response

```json
[
  {
    "mealType": ["lunch"],
    "food_title": "High-Protein Veggie Power Bowl",
    "ingredients": "quinoa, chickpeas, broccoli, red pepper, olive oil, feta, lemon juice"
  }
]
```

---

### 2. Generate Adjusted Recipe

```bash
curl -X POST https://ai-foodhak.com/generate_adjusted_recipe/ \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "123456",
    "food_title": "High-Protein Veggie Power Bowl",
    "mealType": ["lunch"],
    "ingredients": "quinoa, chickpeas, broccoli, red pepper, olive oil, feta, lemon juice"
  }'
```

#### Example JSON Response

```json
[
  {
    "summary": "A balanced lunch bowl packed with plant protein and flavor.",
    "food_title": "High-Protein Veggie Power Bowl",
    "mealType": ["lunch"],
    "servings": "1",
    "ready_in_minutes": "20",
    "ingredients": "quinoa, chickpeas, broccoli, red pepper, olive oil, feta, lemon juice",
    "Calories": "520",
    "Protein": "26.2",
    "Carbohydrates": "68.1",
    "Total Fat": "16.4",
    "Sodium": "440",
    "Saturated Fat": "4.1",
    "Cholesterol": "18",
    "Sugar": "7.4",
    "Calcium": "210",
    "Iron": "4.5",
    "Potassium": "730",
    "Vitamin C": "32",
    "Vitamin E": "3.1",
    "Vitamin D": "0.2",
    "cautions": ["NONE"],
    "diet_labels": ["Vegetarian", "Nut-Free"],
    "instructions": [
      "1. Cook quinoa according to package directions.",
      "2. Sauté broccoli and red pepper in olive oil until just tender.",
      "3. Toss cooked quinoa, chickpeas, veggies, and feta together.",
      "4. Drizzle with lemon juice, season, and serve."
    ],
    "scientific_evidence_support": [
      {
        "description": "High in plant protein and iron. Evidence shows plant-based diets support heart and metabolic health.",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/XXXXXXXX/"
      }
    ],
    "long_description": "A hearty vegetarian lunch bowl designed to hit protein goals without nuts.",
    "main_ingredients": ["quinoa", "chickpeas", "broccoli"],
    "vitamin_mineral_claims": ["High in Iron", "Source of Vitamin C"]
  }
]
```

---

### 3. Health Check

```bash
curl https://ai-foodhak.com/health
```

Response:

```json
{
  "status": "healthy",
  "message": "Recipe Generator Service is up and running."
}
```

---

## 🔒 Authentication

All endpoints except `/health` require an API key:
`Authorization: Bearer <API_KEY>`

* Set in `.env` as `API_KEY`
* Claude and OpenAI keys required for LLM access

---

## 🏷️ Features

* **LLM-powered:** Recipe and adjustment logic handled by Claude 3 Sonnet (fallback: GPT-4o mini)
* **Personalized:** Fully adapts to user’s goals, allergies, restrictions, and preferred macros
* **Transparent:** Nutrition and science justification always included
* **Seamless integration:** All data as valid JSON—no extra text

---

## 📝 Notes

* All requests validated for proper API key.
* If Claude is overloaded, will automatically fall back to OpenAI.
* Service expects all model weights and FAISS indices to be available at startup.
* Supports batch queries and science-backed ingredient retrieval via integrated vector stores.
* Responses strictly follow expected JSON schema for easy UI ingestion.

---
