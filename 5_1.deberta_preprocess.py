import pandas as pd
import numpy as np
import os
import glob
import json
from tqdm import tqdm

# Setting


DATA_DIR = '.'  # Assuming there is a CSV file in the current directory


LABELED_FILE = 'Combined results new cleaned4.csv'
OUTPUT_DIR = '/Home/ajaydev/preprocessed data'  # Change to Linux file system path


os.makedirs(OUTPUT_DIR, exist_ok=True)

# Predefined category system


PREDEFINED_CATEGORIES = {
    'Level1': ['Food', 'Non food'],
    'Level2': {
        'Food': [
            'Beverages', 'Snacks', 'Breakfast', 'Pantry Staples', 
            'Canned & Packaged Foods', 'Cooking & Baking', 
            'Candy & Chocolate', 'Baby Food'
        ]
    },
    'To fluff': {
        'Beverages': [
            'Coffee', 'Tea', 'Water', 'Juice', 'Soda & Soft Drinks', 
            'Sports Drinks', 'Energy Drinks', 'Breakfast Drinks', 
            'Hot Chocolate & Malted Drinks'
        ],
        'Snacks': [
            'Snack Crackers', 'Chips & Pretzels', 'Cookies', 
            'Granola & Nutrition Bars', 'Meat Snacks', 'Nuts & Seeds', 
            'Dried Fruits', 'Snack Cups', 'Popcorn', 'Fruit Snacks'
        ],
        'Breakfast': [
            'Cereal', 'Breakfast & Cereal Bars', 'Syrups & Toppings', 
            'Toaster Pastries', 'Applesauce & Fruit Cups', 
            'Instant Breakfast Drinks'
        ],
        'Pantry Staples': [
            'Canned, Jarred & Packaged Foods', 'Cooking & Baking', 
            'Condiments & Salad Dressings', 'Herbs, Spices & Seasonings', 
            'Sauces, Gravies & Marinades', 'Nut & Seed Butters', 
            'Jams, Jellies & Sweet Spreads', 'Dried Grains & Rice', 
            'Dried Beans, Lentils & Peas'
        ],
        'Canned & Packaged Foods': [
            'Canned Beans', 'Canned & Jarred Tomatoes', 
            'Canned & Jarred Vegetables', 'Canned & Jarred Corn', 
            'Canned & Jarred Fruits', 'Olives, Pickles & Relishes', 
            'Packaged Sloppy Joe Mixes', 'Antipasto', 
            'Packaged Meat, Poultry & Seafood', 'Packaged Meals & Side Dishes'
        ],
        'Cooking & Baking': [
            'Baking Mixes', 'Extracts', 'Baking Syrups, Sugars & Sweeteners', 
            'Lards & Shortenings', 'Frosting, Icing & Decorations', 
            'Baking Chocolates, Carobs & Cocoas', 'Cooking Oils, Vinegars & Sprays', 
            'Dried Fruits & Raisins', 'Baking Flours & Meals', 
            'Baking Leaveners & Yeasts', 'Herbs, Spices & Seasonings', 
            'Cooking & Baking Thickeners', 'Breadcrumbs & Seasoned Coatings'
        ],
        'Candy & Chocolate': [
            'Gummy Candies', 'Chocolate', 'Caramel Candy', 'Hard Candy', 
            'Candy & Chocolate Gifts', 'Licorice Candy', 'Taffy Candy', 
            'Mints', 'Sour Flavored Candies', 'Gum', 'Suckers & Lollipops', 
            'Spicy Sweets', 'Marshmallows'
        ],
        'Baby Food': [
            'Baby Food', 'Baby Cereals', 'Baby Crackers & Biscuits', 
            'Baby Beverages', 'Baby & Toddler Formula', 'Baby Snacks'
        ]
    }
}

# Create category index map


def create_category_indices():
    """
    Create index maps for predefined categories
    """
    category_indices = {
        'Level1': {cat: idx for idx, cat in enumerate(PREDEFINED_CATEGORIES['Level1'])},
        'Level2': {},
        'To fluff': {}
    }
    
    # Level 2 Index Map


    for parent, children in PREDEFINED_CATEGORIES['Level2'].items():
        category_indices['Level2'][parent] = {
            cat: idx for idx, cat in enumerate(children)
        }
    
    # Level 3 index map


    for parent, children in PREDEFINED_CATEGORIES['To fluff'].items():
        category_indices['To fluff'][parent] = {
            cat: idx for idx, cat in enumerate(children)
        }
    
    return category_indices

def preprocess_text(row):
    """
    Combine product_name and search_term to create structured text
    Give higher importance to product_name
    """
    if pd.isna(row['Product name']) or row['Product name'] == 'Search only':
        # If product_name does not exist


        return f"[search] {row['Search term']} [/search]"
    elif pd.isna(row['Search term']) or row['Search term'] == 'Direct access':
        # If there is no search_term


        return f"[product] {row['Product name']} [/product]"
    else:
        # If both information is present


        return f"[product] {row['Product name']} [/PRODUCT] [SEARCH] {row['Search term']} [/search]"

def normalize_category(category, level, parent=None):
    """
    Normalize and validate category values
    """
    if pd.isna(category) or category == '':
        return 'Unknown'
        
    if level == 'Level1':
        # Level 1 normalizes to Food or Non-Food


        if category == 'Food':
            return 'Food'
        else:
            return 'Non food'
    
    # Levels 2 and 3 must have a parent.


    if parent is None:
        return 'Unknown'
    
    # Get list of valid child categories for parent


    if level == 'Level2' and parent in PREDEFINED_CATEGORIES['Level2']:
        valid_categories = PREDEFINED_CATEGORIES['Level2'][parent]
    elif level == 'To fluff' and parent in PREDEFINED_CATEGORIES['To fluff']:
        valid_categories = PREDEFINED_CATEGORIES['To fluff'][parent]
    else:
        return 'Unknown'
    
    # Exact match check


    if category in valid_categories:
        return category
    
    # Find most similar categories (optional)
    # Here it simply returns 'Unknown', 
    # If necessary, you can use string similarity to find the closest category.


    return 'Unknown'

def load_and_preprocess_labeled_data(file_path, category_indices):
    """
    Load and preprocess labeled data
    """
    print(f"Loading labeled data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Select only the columns you need


    if 'Level1 category' in df.columns:
        cols_to_keep = ['Product name', 'Search term', 'Level1 category', 'Level2 category', 'Level3 category']
        df = df[cols_to_keep].copy()
    else:
        print("Warning: No category columns found. Assuming this is prediction data.")
        cols_to_keep = ['Product name', 'Search term']
        df = df[cols_to_keep].copy()
    
    # Text preprocessing


    print("Preprocessing text...")
    df['Text'] = df.apply(preprocess_text, axis=1)
    
    # Category normalization and encoding


    if 'Level1 category' in df.columns:
        # Level 1 Normalization and Encoding


        df['Level1 category'] = df['Level1 category'].apply(lambda x: normalize_category(x, 'Level1'))
        df['Level1 label'] = df['Level1 category'].apply(lambda x: category_indices['Level1'].get(x, -1))
        
        # Level 2 Normalization and Encoding (Food items only)


        df['Level2 label'] = -1  # Default


        food_mask = df['Level1 category'] == 'Food'
        
        # Level 2 processing only for Food items


        for idx, row in df[food_mask].iterrows():
            normalized_l2 = normalize_category(row['Level2 category'], 'Level2', 'Food')
            df.at[idx, 'Level2 category'] = normalized_l2
            
            if normalized_l2 != 'Unknown':
                df.at[idx, 'Level2 label'] = category_indices['Level2']['Food'].get(normalized_l2, -1)
        
        # Level 3 normalization and encoding (only Food items with valid Level 2 categories)


        df['Level3 label'] = -1  # Default


        
        for idx, row in df[food_mask].iterrows():
            level2 = row['Level2 category']
            if level2 != 'Unknown' and level2 in PREDEFINED_CATEGORIES['To fluff']:
                normalized_l3 = normalize_category(row['Level3 category'], 'To fluff', level2)
                df.at[idx, 'Level3 category'] = normalized_l3
                
                if normalized_l3 != 'Unknown':
                    df.at[idx, 'Level3 label'] = category_indices['To fluff'][level2].get(normalized_l3, -1)
    
    return df

def find_empty_category_files():
    """
    Find CSV files with filenames containing 'empty'
    """
    empty_files = glob.glob(os.path.join(DATA_DIR, '*empty*.csv'))
    return empty_files

def preprocess_prediction_data(file_path):
    """
    Preprocessing data to predict
    """
    print(f"Preprocessing prediction data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Select only the columns you need (at least product_name and search_term are required)


    required_cols = ['Product name', 'Search term']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns in {file_path}. Need at least {required_cols}")
    
    df = df[required_cols].copy()
    
    # Text preprocessing


    df['Text'] = df.apply(preprocess_text, axis=1)
    
    return df

def main():
    # 1. Create a category index map


    category_indices = create_category_indices()
    
    # 2. Save category system


    with open(os.path.join(OUTPUT_DIR, 'Predefined categories.json'), 'W') as f:
        json.dump(PREDEFINED_CATEGORIES, f, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, 'Category indices.json'), 'W') as f:
        json.dump(category_indices, f, indent=2)
    
    # 3. Load and preprocess labeled data


    labeled_file_path = os.path.join(DATA_DIR, LABELED_FILE)
    labeled_df = load_and_preprocess_labeled_data(labeled_file_path, category_indices)
    
    # 4. Save preprocessed training data


    train_output_path = os.path.join(OUTPUT_DIR, 'Preprocessed train.csv')
    labeled_df.to_csv(train_output_path, index=False)
    print(f"Preprocessed training data saved to {train_output_path}")
    
    # 5. Find data (empty files) to predict


    empty_files = find_empty_category_files()
    print(f"found {len(empty_files)} files for prediction")
    
    # 6. Preprocess each prediction file


    for empty_file in empty_files:
        file_name = os.path.basename(empty_file)
        try:
            pred_df = preprocess_prediction_data(empty_file)
            pred_output_path = os.path.join(OUTPUT_DIR, f'preprocessed{file_name}')
            pred_df.to_csv(pred_output_path, index=False)
            print(f"Preprocessed prediction data saved to {pred_output_path}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    # 7. Statistical output


    print("\nPreprocessing Summary:")
    print(f"Training samples: {len(labeled_df)}")
    
    # Statistics by level


    if 'Level1 category' in labeled_df.columns:
        print(f"Level 1 distribution:")
        level1_counts = labeled_df['Level1 category'].value_counts()
        for cat, count in level1_counts.items():
            print(f"  {cat}: {count} ({count/len(labeled_df)*100:.2f}%)")
        
        # Level 2 statistics for Food items


        food_df = labeled_df[labeled_df['Level1 category'] == 'Food']
        if len(food_df) > 0:
            print(f"\nLevel 2 distribution (Food items only, total: {len(food_df)}):")
            level2_counts = food_df['Level2 category'].value_counts()
            for cat, count in level2_counts.items():
                if cat != 'Unknown':
                    print(f"  {cat}: {count} ({count/len(food_df)*100:.2f}%)")
            
            # Level 3 Statistics by Valid Level 2 Category


            print(f"\nLevel 3 distribution by Level 2 category:")
            for level2 in PREDEFINED_CATEGORIES['To fluff'].keys():
                level2_df = food_df[food_df['Level2 category'] == level2]
                if len(level2_df) > 0:
                    print(f"  {level2} (total: {len(level2_df)}):")
                    level3_counts = level2_df['Level3 category'].value_counts()
                    for cat, count in level3_counts.items():
                        if cat != 'Unknown':
                            print(f"    {cat}: {count} ({count/len(level2_df)*100:.2f}%)")

if __name__ == "Main":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()