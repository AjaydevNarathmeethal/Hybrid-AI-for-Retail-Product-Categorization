import streamlit as st
import torch
import os
import json
import pandas as pd
import numpy as np
import requests
import time
import random
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DebertaV2Config

# Settings

MAX_LEN = 256
MODEL_PATH = "../models/best model step 14000"  # Path to model in parent directory

CATEGORIES_PATH = "./predefined categories.json"  # Category information file

INDICES_PATH = "./category indices.json"  # Category index file

DEVICE = torch.device('Cuda' if torch.cuda.is_available() else 'Cpu')
FP16 = torch.cuda.is_available()  # Use FP16 if GPU is available


# Page setup

st.set_page_config(
    page_title="Food Categorizer",
    page_icon="🍔",
    layout="Wide",
    initial_sidebar_state="Expanded"
)

# App header

st.title("🍔 Food Categorizer")
st.markdown("This app analyzes text input to classify Food/Non-Food items and predicts detailed categories for Food items.")

# Load category information function

@st.cache_resource
def load_category_info():
    st.info("Loading category information...")
    
    # Load category information from files

    if os.path.exists(CATEGORIES_PATH) and os.path.exists(INDICES_PATH):
        # Load from files

        with open(CATEGORIES_PATH, 'R') as f:
            categories = json.load(f)
        
        with open(INDICES_PATH, 'R') as f:
            indices = json.load(f)
    else:
        st.error(f"Category files not found: {CATEGORIES_PATH} Or {INDICES_PATH}")
        categories = {
            "Level1": ["Food", "Non food"],
            "Level2": {"Food": ["Beverages", "Snacks", "Breakfast"]},
            "To fluff": {"Beverages": ["Coffee", "Tea", "Water"]}
        }
        indices = {
            "Level1": {"Food": 0, "Non food": 1},
            "Level2": {"Food": {"Beverages": 0, "Snacks": 1, "Breakfast": 2}},
            "To fluff": {"Beverages": {"Coffee": 0, "Tea": 1, "Water": 2}}
        }
    
    # Create index-to-category mapping (reverse mapping)

    idx_to_category = {}
    
    # Level 1 processing

    idx_to_category['Level1'] = {int(v): k for k, v in indices['Level1'].items()}
    
    # Level 2 processing

    idx_to_category['Level2'] = {}
    for parent, children in indices['Level2'].items():
        for child, idx in children.items():
            idx_to_category['Level2'][int(idx)] = child
    
    # Level 3 processing

    idx_to_category['To fluff'] = {}
    for parent, children in indices['To fluff'].items():
        for child, idx in children.items():
            idx_to_category['To fluff'][int(idx)] = child
    
    return categories, indices, idx_to_category

# Hierarchical classification model class

class HierarchicalDebertaClassifier(torch.nn.Module):
    def __init__(self, model_path, num_level1, num_level2, num_level3):
        super(HierarchicalDebertaClassifier, self).__init__()
        self.num_level1 = num_level1
        self.num_level2 = num_level2
        self.num_level3 = num_level3
        
        # DeBERTa configuration and loading

        config = DebertaV2Config.from_pretrained(model_path)
        config.output_hidden_states = True
        
        # Load model

        self.deberta = DebertaV2ForSequenceClassification.from_pretrained(
            model_path,
            config=config
        )
        
        # Additional classification heads

        hidden_size = self.deberta.config.hidden_size
        self.dropout = torch.nn.Dropout(0.2)
        self.level2_classifier = torch.nn.Linear(hidden_size, self.num_level2)
        self.level3_classifier = torch.nn.Linear(hidden_size, self.num_level3)
    
    def forward(self, input_ids, attention_mask):
        # Get DeBERTa outputs

        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Level 1 classification (Food/Non-Food)

        level1_logits = outputs.logits
        
        # Get hidden states

        hidden_states = outputs.hidden_states[-1][:, 0, :]
        hidden_states = self.dropout(hidden_states)
        
        # Level 2, 3 classification

        level2_logits = self.level2_classifier(hidden_states)
        level3_logits = self.level3_classifier(hidden_states)
        
        return level1_logits, level2_logits, level3_logits

# Model loading function

@st.cache_resource
def load_model(model_path, num_level1, num_level2, num_level3):
    try:
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
        model = HierarchicalDebertaClassifier(model_path, num_level1, num_level2, num_level3)
        model.to(DEVICE)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.warning("Switching to demo mode. Random results will be generated if no model is available.")
        return None, None

# DeBERTa model prediction function

def predict_with_deberta(text, tokenizer, model, idx_to_category):
    if not tokenizer or not model:
        # Demo mode: Return random results without a model

        import random
        is_food = random.choice([True, False])
        level1 = "Food" if is_food else "Non food"
        
        if is_food:
            level2_categories = ["Beverages", "Snacks", "Breakfast", "Pantry Staples", 
                               "Canned & Packaged Foods", "Cooking & Baking", 
                               "Candy & Chocolate", "Baby Food"]
            level2 = random.choice(level2_categories)
            
            # Select level3 category corresponding to level2

            if level2 in idx_to_category['To fluff']:
                level3_idx = random.choice(list(idx_to_category['To fluff'][level2].values()))
                level3 = next((cat for cat, idx in idx_to_category['To fluff'][level2].items() if idx == level3_idx), "Other")
            else:
                level3 = "Other"
        else:
            level2 = "N/a"
            level3 = "N/a"
        
        # Indicate that it's demo mode

        st.warning("Demo mode active: Random results are being generated because no actual model is available.")
        
        return {
            'Level1 category': level1,
            'Level2 category': level2,
            'Level3 category': level3,
            'Confidence': random.uniform(0.7, 0.95)
        }
    
    # Actual model use logic

    try:
        with st.spinner("Analyzing with DeBERTa model..."):
            # Tokenizing

            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding='Max length',
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='Pt'
            )
            
            input_ids = encoding['Input ids'].to(DEVICE)
            attention_mask = encoding['Attention mask'].to(DEVICE)
            
            # Prediction

            with torch.no_grad():
                if FP16 and torch.cuda.is_available():
                    with torch.amp.autocast('Cuda'):  # Using torch.amp.autocast('cuda') instead of torch.cuda.amp.autocast()

                        level1_logits, level2_logits, level3_logits = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                else:
                    level1_logits, level2_logits, level3_logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
            
            # Convert prediction values

            level1_preds = torch.argmax(level1_logits, dim=1).cpu().numpy()
            level2_preds = torch.argmax(level2_logits, dim=1).cpu().numpy()
            level3_preds = torch.argmax(level3_logits, dim=1).cpu().numpy()
            
            # Calculate confidence

            level1_probs = torch.softmax(level1_logits, dim=1).cpu().numpy()
            level1_confidence = level1_probs[0, level1_preds[0]]
            
            # Find Food category index

            food_idx = None
            for idx, category in idx_to_category['Level1'].items():
                if category == 'Food':
                    food_idx = idx
                    break
            
            # Generate results

            level1_category = idx_to_category['Level1'][level1_preds[0]]
            
            # Use level2, level3 predictions only if Food

            is_food = (level1_preds[0] == food_idx)
            
            if is_food:
                level2_category = idx_to_category['Level2'][level2_preds[0]]
                
                # Level3 is dependent on level2, so find from the subcategories of that level2

                level3_categories = idx_to_category['To fluff'].get(level2_category, {})
                
                # Find the closest match from all level3 indices

                if level3_categories:
                    level3_pred = level3_preds[0]
                    matched = False
                    
                    # Try to find an exact matching index

                    for cat, idx in level3_categories.items():
                        if idx == level3_pred:
                            level3_category = cat
                            matched = True
                            break
                    
                    # If no exact match, choose the closest value

                    if not matched:
                        level3_indices = list(level3_categories.values())
                        closest_idx = min(level3_indices, key=lambda x: abs(x - level3_pred))
                        for cat, idx in level3_categories.items():
                            if idx == closest_idx:
                                level3_category = cat
                                break
                else:
                    level3_category = "Other"
            else:
                level2_category = "N/a"
                level3_category = "N/a"
            
            return {
                'Level1 category': level1_category,
                'Level2 category': level2_category,
                'Level3 category': level3_category,
                'Confidence': float(level1_confidence)
            }
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return {
            'Level1 category': "Error",
            'Level2 category': "Error",
            'Level3 category': "Error",
            'Confidence': 0.0
        }

# Ollama API-based level 2 classification function (modified: import streamlit in the function)

def classify_with_ollama(search_terms, product_names, batch_size=3, use_cuda=True, model="Llama3:latest", debug_mode=False):
    """
    Optimized Ollama API-based classification function
    
    Args:
        search_terms: List of search terms
        product_names: List of product names
        batch_size: Batch size (2-3 is appropriate for RTX 3070)
        use_cuda: Whether to use CUDA
        model: Ollama model name
        debug_mode: Debug mode flag
    
    Returns:
        List of classification results
    """
    # Import streamlit in the function

    import streamlit as st
    
    # Single string handling -convert to list

    if isinstance(search_terms, str):
        search_terms = [search_terms]
    if isinstance(product_names, str):
        product_names = [product_names]
    
    # Length check and adjustment

    if len(product_names) < len(search_terms):
        product_names = product_names + [product_names[-1]] * (len(search_terms) - len(product_names))
    elif len(product_names) > len(search_terms):
        product_names = product_names[:len(search_terms)]
    
    # Cache system implementation (prevent duplicate requests for the same product)

    cache = {}
    cache_hits = 0  # Function-level variable

    
    # Result storage list

    results = []
    
    # Session reuse (API connection optimization)

    session = requests.Session()
    
    # Streamlit progress display (text only)

    progress_text = "Classifying with Ollama API..."
    status = st.empty()
    status.text(progress_text)
    
    # Batch processing

    total_batches = len(search_terms) // batch_size + (1 if len(search_terms) % batch_size > 0 else 0)
    
    for i in range(0, len(search_terms), batch_size):
        batch_st = search_terms[i:i+batch_size]
        batch_pn = product_names[i:i+batch_size]
        
        batch_results = []
        
        # Batch progress update

        status.text(f"{progress_text} ({i+1}/{len(search_terms)} Items)")
        
        # Process each item in the batch

        for st, pn in zip(batch_st, batch_pn):
            # Cache check (prevent duplicate requests)

            cache_key = f"{st}_{pn}"
            if cache_key in cache:
                batch_results.append(cache[cache_key])
                cache_hits += 1
                continue
            
            # Request interval optimization

            time.sleep(random.uniform(0.15, 0.3))
            
            # Use original prompt from the code

            prompt = f"""Task: Classify the following product into a Food Level 2 category based on product information.

Search Term: {st} Product Name: {pn}

Level 2 Food Categories:

Beverages: All drinkable liquid food products, including alcoholic beverages • Examples: Coffee, tea, juice, water, soda, energy drinks, sports drinks, bubble tea (boba), beer, wine, milk, yogurt drinks • Note: Coffee syrups, concentrates, and flavorings for drinks belong here, not in Pantry Staples

Snacks: Lightweight foods typically eaten between meals, excluding candy and chocolate • Examples: Chips, crackers, pretzels, popcorn, cookies, nuts, granola bars, trail mix, fruit snacks • Note: Cookies belong here even if they contain chocolate; marshmallows used as toppings (e.g., mini marshmallows) belong in Cooking & Baking

Breakfast: Foods primarily associated with morning meals, excluding drinks • Examples: Cereal, oatmeal, pancake mix, waffles, breakfast bars, bread, bagels • Note: Maple syrup belongs in Pantry Staples; coffee, tea, and milk belong in Beverages

Pantry Staples: Basic cooking ingredients and shelf-stable foods used in preparation • Examples: Cooking oils, spices, flour, sugar, rice, pasta, peanut butter, jams, jellies, maple syrup, raw nuts, yeast • Note: Raw cooking ingredients go here; baking mixes belong in Cooking & Baking

Canned & Packaged Foods: Ready-to-eat or ready-to-heat packaged foods, including frozen items • Examples: Canned soups, canned vegetables, canned fruits, olives, pickles, packaged meals, frozen dinners, frozen meats, fresh produce (fruits, vegetables) • Note: All canned goods, soups, prepared meals, and frozen foods (e.g., shrimp, pizza rolls) belong here

Cooking & Baking: Specialized ingredients specifically for cooking and baking • Examples: Baking mixes, extracts, frosting, cake decorations, baking chocolate, mini marshmallows (for baking), gelatin mixes • Note: General pantry items like flour and sugar belong in Pantry Staples

Candy & Chocolate: Sweet confectionery items, excluding baked goods • Examples: Chocolate bars, gummy candy, hard candy, caramels, lollipops, mints, chewing gum • Note: Cookies and baked goods belong in Snacks, even if they contain chocolate

Baby Food: Food products specifically formulated for infants and young children • Examples: Baby purees, infant formula, baby cereal, teething biscuits • Note: Regular foods that adults also eat do NOT belong here

Non-Food: Items that are not human food products • Examples: Pet food, kitchen appliances, cleaning products • Note: Use this category for non-edible items

Important Classification Guidelines:

Soups belong in "Canned & Packaged Foods", not Breakfast
Tea, coffee, and drink flavorings (e.g., coffee syrup, concentrate) always go in "Beverages", not Pantry Staples
Olives and pickles belong in "Canned & Packaged Foods"
Bread and bagels belong in "Breakfast"
Boba/bubble tea ingredients go in "Beverages"
Fresh produce (fruits, vegetables) and frozen foods (e.g., shrimp, pizza rolls) go in "Canned & Packaged Foods"
Cookies belong in "Snacks", even if they contain chocolate (e.g., "Chips Ahoy with Reese's")
Chewing gum belongs in "Candy & Chocolate", not Snacks
If the product is a kitchen appliance (e.g., waffle maker, air fryer), classify as "Non-Food"
If search_term is "direct_access", use only product_name for classification
If the product name indicates a frozen item (e.g., "frozen shrimp", "frozen dinner"), classify as "Canned & Packaged Foods"

Few-shot Examples:

Product: "progresso traditional chicken noodle soup ready to serve 19 oz 4 pack" Correct category: Canned & Packaged Foods
Product: "wonder bread family loaf pack of 2" Correct category: Breakfast
Product: "heavenly tea leaves 9 flavor variety pack loose leaf tea sampler" Correct category: Beverages
Product: "byzantine pitted olive mix country 5 pound" Correct category: Canned & Packaged Foods
Product: "rainbow boba tea real tapioca pearls ready in 5 minutes" Correct category: Beverages
Product: "butterscotch buttons hard candy 1 lb" Correct category: Candy & Chocolate
Product: "little debbie big pack oatmeal creme pies 32 oz" Correct category: Snacks
Product: "authentic asia thai basil stir fried beef meal 10 oz frozen dinner" Correct category: Canned & Packaged Foods
Product: "chips ahoy cookies with reeses peanut butter cups family size 1425 oz" Correct category: Snacks
Product: "great value mini marshmallows 10 oz" Correct category: Cooking & Baking
Product: "trident sugar free gum spearmint flavor 3 packs 42 pieces total" Correct category: Candy & Chocolate
Product: "great value frozen cooked extra small peeled deveined tailoff shrimp 12 oz" Correct category: Canned & Packaged Foods
Product: "javy coffee concentrate cold brew coffee 35 servings caramel brulee" Correct category: Beverages
Product: "chefman rotating belgian waffle maker 180 flip iron" Correct category: Non-Food
Product: "pedigree complete nutrition adult dry dog food grilled steak vegetable flavor 18 lb bag" Correct category: Non-Food

Return ONLY the level2 category name as a single word or phrase. No explanation or additional text."""
            
            # Retry logic optimization (exponential backoff)

            max_retries = 2
            category_text = "Error"  # Default value

            
            for attempt in range(max_retries):
                try:
                    # API request log

                    if debug_mode:
                        print(f"API Request: {st} | {pn} (attempt {attempt+1})")
                    
                    # Ollama API call

                    response = session.post(
                        "Http://localhost:11434/api/generate",
                        json={
                            "Model": model,
                            "Prompt": prompt,
                            "Stream": False,
                            "Temperature": 0.05,  # Lower temperature (better consistency)

                            "Max tokens": 15,  # Fewer tokens

                        },
                        timeout=25  # Timeout

                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        category_text = result.get('Response', '').strip()
                        
                        # Response refinement

                        category_text = category_text.replace('"', '').strip()
                        
                        # Category validation (check if it's a valid category)

                        valid_categories = [
                            "Beverages", "Snacks", "Breakfast", "Pantry Staples",
                            "Canned & Packaged Foods", "Cooking & Baking",
                            "Candy & Chocolate", "Baby Food", "Non food"
                        ]
                        
                        # Efficient category matching

                        category_text_lower = category_text.lower()
                        found = False
                        
                        # Try exact matching first

                        for valid_cat in valid_categories:
                            valid_cat_lower = valid_cat.lower()
                            if valid_cat_lower == category_text_lower:
                                category_text = valid_cat  # Change to correct case

                                found = True
                                break
                        
                        # Try partial matching

                        if not found:
                            for valid_cat in valid_categories:
                                valid_cat_lower = valid_cat.lower()
                                if valid_cat_lower in category_text_lower or category_text_lower in valid_cat_lower:
                                    category_text = valid_cat
                                    found = True
                                    break
                        
                        # Handle not found

                        if not found:
                            category_text = "Unknown"
                        
                        # Save result to cache

                        cache[cache_key] = category_text
                        
                        if debug_mode:
                            print(f"Classification result: {category_text}")
                        
                        break  # End retry loop on success

                    else:
                        if debug_mode:
                            print(f"API request failed (status code: {response.status_code})")
                        category_text = "Error"
                        
                        # Exponential backoff

                        if attempt < max_retries - 1:
                            backoff_time = (2 ** attempt) * 1.0
                            time.sleep(backoff_time)
                
                except Exception as e:
                    if debug_mode:
                        print(f"error: {str(e)}")
                    category_text = "Error"
                    if attempt < max_retries - 1:
                        backoff_time = (2 ** attempt) * 1.0
                        time.sleep(backoff_time)
            
            batch_results.append(category_text)
        
        results.extend(batch_results)
        
        # GPU memory cleanup between batches (conditional execution)

        if i % (batch_size * 5) == 0 and use_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Shorter rest between batches

        time.sleep(0.3)
    
    # Clear progress display

    status.empty()
    
    if debug_mode:
        print(f"Cache hits: {cache_hits} ({cache_hits/len(search_terms)*100:.1f}% of all requests)")
    
    return results

# Single item Ollama classification function

def classify_single_item_with_ollama(text, product_name="", model="Llama3:latest", debug_mode=False):
    """
    Single item Ollama API classification function
    """
    results = classify_with_ollama(
        search_terms=text, 
        product_names=product_name if product_name else text,
        batch_size=1,
        use_cuda=torch.cuda.is_available(),
        model=model,
        debug_mode=debug_mode
    )
    
    if results and len(results) > 0:
        return results[0]
    else:
        return "Classification Error"

# Main app function

def main():
    # Session state initialization

    if 'Debug mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    # Sidebar

    st.sidebar.title("Settings")
    
    # Debug mode toggle

    st.session_state.debug_mode = st.sidebar.checkbox("Debug Mode", value=st.session_state.debug_mode)
    
    # Ollama model selection

    ollama_model = st.sidebar.selectbox(
        "Select Llama Model",
        ["Llama3:latest", "Shiny: lizard", "Shiny: 70B"],
        index=0
    )
    
    # API server status check

    if st.sidebar.button("Check Ollama API Server Status"):
        try:
            response = requests.get("Http://localhost:11434/api/tags")
            if response.status_code == 200:
                st.sidebar.success("✅ Ollama API server is running")
                models = response.json().get("Models", [])
                if models:
                    st.sidebar.write("Available models:")
                    for model in models:
                        st.sidebar.write(f" {model['Name']}")
            else:
                st.sidebar.error("❌ Ollama API server response error")
        except Exception as e:
            st.sidebar.error(f"❌ Ollama API server connection failed: {str(e)}")
    
    # Load category information

    categories, indices, idx_to_category = load_category_info()
    
    # Display model information (debug mode)

    if st.session_state.debug_mode:
        st.sidebar.subheader("Category Information")
        st.sidebar.write(f"Level 1 categories: {len(categories['Level1'])}")
        st.sidebar.write(f"Level 2 categories (Food): {len(categories['Level2']['Food'])}")
        st.sidebar.write(f"Level 3 categories: {sum(len(cats) for cats in categories['To fluff'].values())}")
        st.sidebar.write(f"Model path: {MODEL_PATH}")
        st.sidebar.write(f"Category files: {CATEGORIES_PATH}, {INDICES_PATH}")
    
    # Calculate number of categories by level

    num_level1 = len(categories['Level1'])
    num_level2 = len(categories['Level2']['Food'])
    num_level3 = sum(len(children) for parent, children in categories['To fluff'].items())
    
    # Load model

    tokenizer, model = load_model(MODEL_PATH, num_level1, num_level2, num_level3)
    
    # User input

    st.subheader("Product Information Input")
    col1, col2 = st.columns(2)
    
    with col1:
        user_input = st.text_area("Enter product description or search term:", height=100,
                               placeholder="Example: progresso traditional chicken noodle soup ready to serve 19 oz 4 pack")
    
    with col2:
        product_name = st.text_area("Product Name (Optional):", height=100, 
                                placeholder="Enter product name if different from search term")
    
    # Analyze button

    if st.button("Analyze", type="Primary", use_container_width=True):
        if not user_input.strip():
            st.error("Please enter a product description or search term!")
            return
        
        # Step-by-step prediction UI

        step1_container = st.container()
        step2_container = st.container()
        
        with step1_container:
            st.subheader("🥑 Step 1: Food/Non-Food Classification")
            
            # Level 1 prediction via DeBERTa

            level1_result = predict_with_deberta(user_input, tokenizer, model, idx_to_category)
            
            # Display results

            col1, col2, col3 = st.columns([2, 2, 3])
            with col1:
                st.metric("Classification Result", level1_result['Level1 category'])
            with col2:
                st.metric("Confidence", f"{level1_result['Confidence']:.2%}")
            with col3:
                if level1_result['Level1 category'] == 'Food':
                    st.success("✅ Classified as Food item. Proceeding to detailed classification.")
                else:
                    st.info("ℹ️ Classified as Non-Food item. Analysis complete.")
        
        # Proceed to Level 2 classification only if Food

        if level1_result['Level1 category'] == 'Food':
            with step2_container:
                st.subheader("🍕 Step 2: Food Category Classification")
                
                # Analysis method selection

                method = st.radio(
                    "Select classification method:",
                    ["DeBERTa Model", "Llama3 API"],
                    horizontal=True,
                    index=1
                )
                
                if method == "DeBERTa Model":
                    # Use DeBERTa results

                    level2_category = level1_result['Level2 category']
                    level3_category = level1_result['Level3 category']
                    
                    # Display completion message

                    st.success(f"Level 2 Category: {level2_category}")
                
                else:  # Use Llama3 API
                    # Level 2 prediction via Llama3

                    level2_category = classify_single_item_with_ollama(
                        user_input, 
                        product_name if product_name else user_input,
                        model=ollama_model,
                        debug_mode=st.session_state.debug_mode
                    )
                    
                    # Display completion message

                    if "Error" in level2_category or "Unknown" in level2_category:
                        st.error(f"error: {level2_category}")
                    else:
                        st.success(f"Level 2 Category: {level2_category}")
                
                # Category descriptions

                level2_desc = {
                    "Beverages": "All drinkable liquid food products (coffee, tea, juice, water, soda, energy drinks, beer, wine, etc.)",
                    "Snacks": "Lightweight foods typically eaten between meals (chips, crackers, nuts, popcorn, cookies, etc.)",
                    "Breakfast": "Foods primarily associated with morning meals (cereal, oatmeal, pancake mix, waffles, breakfast bars, bread, bagels, etc.)",
                    "Pantry Staples": "Basic cooking ingredients and shelf-stable foods (cooking oils, spices, flour, sugar, rice, pasta, etc.)",
                    "Canned & Packaged Foods": "Ready-to-eat or ready-to-heat packaged foods (canned soups, vegetables, fruits, olives, pickles, frozen foods, etc.)",
                    "Cooking & Baking": "Specialized ingredients for cooking and baking (baking mixes, extracts, frosting, cake decorations, baking chocolate, etc.)",
                    "Candy & Chocolate": "Sweet confectionery items (chocolate bars, gummy candy, hard candy, caramels, gum, etc.)",
                    "Baby Food": "Food products specifically for infants and young children (baby purees, infant formula, baby cereal, etc.)"
                }
                
                if method == "Llama3 API" and level2_category in level2_desc:
                    with st.expander("View Category Description"):
                        st.write(level2_desc[level2_category])
                        
                # Final result summary

                st.subheader("📋 Analysis Summary")
                result_df = pd.DataFrame({
                    "Classification Level": ["Level 1", "Level 2"],
                    "Category": [level1_result['Level1 category'], 
                              level2_category if level1_result['Level1 category'] == 'Food' else "N/a"],
                    "Method": ["debandhim", 
                              method if level1_result['Level1 category'] == 'Food' else "N/a"]
                })
                st.table(result_df)
        
        # Complete processing message

        st.success("Analysis completed! 👍")
        
        # Save analysis history (optional)

        if st.checkbox("Save this result to analysis history"):
            if 'History' not in st.session_state:
                st.session_state.history = []
            
            timestamp = time.strftime("%y %m%d %h:%m:%s")
            st.session_state.history.append({
                "Timestamp": timestamp,
                "Input": user_input,
                "Product name": product_name,
                "Level1": level1_result['Level1 category'],
                "Level2": level2_category if level1_result['Level1 category'] == 'Food' else "N/a"
            })
            
            st.success("Result saved to analysis history.")
    
    # Display analysis history

    if 'History' in st.session_state and st.session_state.history:
        with st.expander("View Analysis History"):
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df)
            
            if st.button("Clear History"):
                st.session_state.history = []
                st.experimental_rerun()

# Run app

if __name__ == "Main":
    main()