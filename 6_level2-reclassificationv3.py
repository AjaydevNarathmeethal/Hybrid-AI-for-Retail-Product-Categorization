import pandas as pd
import numpy as np
import os
import time
import json
import requests
from tqdm import tqdm
import random
from datetime import datetime
import gc
import torch
import threading

def classify_with_ollama(search_terms, product_names, batch_size=3, use_cuda=True, model="Llama3:latest"):
    """
    Optimized Olama API-based classification function
    
    Args:
        search_terms: list of search terms
        product_names: list of product names
        batch_size: Batch size (2-3 is appropriate for RTX 3070)
        use_cuda: Whether to use CUDA
        model: Ollama model name
    
    Returns:
        Classification result list
    """
    # Implement a cache system (avoid duplicate requests for the same product)


    cache = {}
    cache_hits = 0  # Variables defined at the function level


    
    # System resource monitoring functions


    def check_resources():
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        gpu_memory_used = 0
        if use_cuda and torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated()
            gpu_total_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_used = (gpu_memory_allocated / gpu_total_memory) * 100
        
        print(f"System Status: CPU {cpu_percent}%, memory {memory_percent}%, GPU memory {gpu_memory_used:.1f}%")
        
        is_critical = (cpu_percent > 90 or memory_percent > 90 or gpu_memory_used > 90)
        is_warning = (cpu_percent > 75 or memory_percent > 75 or gpu_memory_used > 75)
        
        return is_critical, is_warning, gpu_memory_used
    
    # Result storage list


    results = []
    
    # Initial resource check


    is_critical, is_warning, _ = check_resources()
    if is_critical:
        print("⚠️ System resources are at critical level. It starts after waiting 20 seconds.")
        time.sleep(20)
        if use_cuda:
            torch.cuda.empty_cache()
    elif is_warning:
        print("⚠️ System resources are high. It starts after waiting 10 seconds.")
        time.sleep(10)
    
    # Session reuse (optimizing API connections)


    session = requests.Session()
    
    # batch processing


    total_batches = len(search_terms) // batch_size + (1 if len(search_terms) % batch_size > 0 else 0)
    
    with tqdm(total=len(search_terms), desc="Classification progress") as pbar:
        for i in range(0, len(search_terms), batch_size):
            # Periodic resource inspection (expanded to every 15th batch -inspection frequency reduced)


            if i > 0 and i % (batch_size * 15) == 0:
                is_critical, is_warning, gpu_usage = check_resources()
                if is_critical:
                    print(f"\n⚠️ System resources are at critical level. Rest 20 seconds. (Progress: {i}/{len(search_terms)})")
                    time.sleep(20)
                    if use_cuda:
                        torch.cuda.empty_cache()
                elif is_warning:
                    print(f"\n⚠️ System resources are high. Rest 5 seconds. (Progress: {i}/{len(search_terms)})")
                    time.sleep(5)
                    if use_cuda:
                        torch.cuda.empty_cache()
                        
                # Dynamic batch sizing (based on GPU utilization)


                if gpu_usage < 30 and batch_size < 4:
                    batch_size += 1
                    print(f"GPU utilization is low: change batch size to {batch_size}increases to")
                elif gpu_usage > 80 and batch_size > 1:
                    batch_size -= 1
                    print(f"High GPU utilization: Reduce batch size to {batch_size}decrease to")
            
            batch_st = search_terms[i:i+batch_size]
            batch_pn = product_names[i:i+batch_size]
            
            batch_results = []
            
            # Process each item in a batch


            for st, pn in zip(batch_st, batch_pn):
                # Check cache (avoid duplicate requests)


                cache_key = f"{st}_{pn}"
                if cache_key in cache:
                    batch_results.append(cache[cache_key])
                    cache_hits += 1  # Update function level variables (no need for nonlocal)


                    continue
                
                # Optimize delay time between requests (0.2~0.4 seconds → 0.15~0.3 seconds)


                time.sleep(random.uniform(0.15, 0.3))
                
                # Updated prompt


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
                
                # Optimized retry logic (with exponential backoff)


                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        # Simplified log output (reduce log when making API requests)


                        if attempt == 0:
                            print(f"\nAPI request: {st} | {pn}")
                        else:
                            print(f"Retry {attempt+1}/{max_retries}")
                        
                        # Ollama API call optimization


                        response = session.post(  # Session Reuse


                            "Http://localhost:11434/api/generate",
                            json={
                                "Model": model,  # Use function parameters


                                "Prompt": prompt,
                                "Stream": False,
                                "Temperature": 0.05,  # Lower Temperatures (Better Consistency)


                                "Max tokens": 15,  # smaller number of tokens


                            },
                            timeout=25  # Slightly adjusted timeout


                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            category_text = result.get('Response', '').strip()
                            
                            # Response refinement (efficient processing)


                            category_text = category_text.replace('"', '').strip()
                            
                            # Category validation (ensures it is a valid category)


                            valid_categories = [
                                "Beverages", "Snacks", "Breakfast", "Pantry Staples",
                                "Canned & Packaged Foods", "Cooking & Baking",
                                "Candy & Chocolate", "Baby Food", "Non food"
                            ]
                            
                            # Efficient category matching (pre-lowercase conversion)


                            category_text_lower = category_text.lower()
                            found = False
                            
                            # Try accurate matching first


                            for valid_cat in valid_categories:
                                valid_cat_lower = valid_cat.lower()
                                if valid_cat_lower == category_text_lower:
                                    category_text = valid_cat  # Change to correct case


                                    found = True
                                    break
                            
                            # Partial matching attempt


                            if not found:
                                for valid_cat in valid_categories:
                                    valid_cat_lower = valid_cat.lower()
                                    if valid_cat_lower in category_text_lower or category_text_lower in valid_cat_lower:
                                        category_text = valid_cat
                                        found = True
                                        break
                            
                            # Handle if not found


                            if not found:
                                category_text = "Unknown"
                            
                            # Store results in cache


                            cache[cache_key] = category_text
                            print(f"Classification result: {category_text}")
                            
                            break  # End retry loop on success


                        else:
                            print(f"API request failed (status code: {response.status_code})")
                            category_text = "Error"
                            
                            # Exponential backoff (wait 1 second for first failure, 2 seconds for second)


                            if attempt < max_retries - 1:
                                backoff_time = (2 ** attempt) * 1.0  # 1 second, 2 seconds...


                                time.sleep(backoff_time)
                    
                    except Exception as e:
                        print(f"error: {str(e)}")
                        category_text = "Error"
                        if attempt < max_retries - 1:
                            backoff_time = (2 ** attempt) * 1.0
                            time.sleep(backoff_time)
                
                batch_results.append(category_text)
            
            results.extend(batch_results)
            pbar.update(len(batch_st))
            
            # GPU memory cleanup between batches (conditional execution -not every batch)


            if i % (batch_size * 5) == 0 and use_cuda:  # Organized every 5 batches


                torch.cuda.empty_cache()
            
            # Shorter breaks between batches


            time.sleep(0.3)  # Decreased from 0.5 seconds to 0.3 seconds


    
    print(f"Cache hit: {cache_hits} (of the entire request) {cache_hits/len(search_terms)*100:.1f}%)")
    return results

def save_checkpoint_async(df, filename):
    """Asynchronous checkpoint storage function"""
    df_copy = df.copy()  # Copy data (for thread safety)


    
    def save_func():
        try:
            # Improved I/O speed by storing only necessary columns


            essential_cols = ['Search term', 'Product name', 'Level1 category', 'Level2 category']
            cols_to_save = [col for col in essential_cols if col in df_copy.columns]
            df_copy[cols_to_save].to_csv(filename, index=False)
            print(f"Checkpoint saved: {filename}")
        except Exception as e:
            print(f"Failed to save checkpoint: {str(e)}")
    
    # Execute save in separate thread


    thread = threading.Thread(target=save_func)
    thread.daemon = True  # Set to daemon thread (automatically terminates when main exits)


    thread.start()
    
    return thread

def process_food_classification(data_path, output_path=None, batch_size=3, chunk_size=500):
    """
    Level 2 food classification overall processing function (speed optimization)
    
    Args:
        data_path: Input data file path
        output_path: Output file path (automatically created if None)
        batch_size: batch size
        chunk_size: Chunk size (checkpoint storage interval)
    """
    print("Loading data...")
    
    # Same data loading part as previous code...


    for encoding in ['Utf 8', 'Subs49', 'Latin1']:
        try:
            df = pd.read_csv(data_path, encoding=encoding)
            print(f"{encoding} Load success with encoding")
            break
        except Exception as e:
            print(f"{encoding} Load failure with encoding: {str(e)}")
    else:
        try:
            # Last attempt: Error Ignore option


            df = pd.read_csv(data_path, encoding='Latin1', on_bad_lines='Skip')
            print("Successful loading with Latin-1 encoding + error ignore option")
        except Exception as e:
            print(f"All load attempts failed: {str(e)}")
            return None
    
    print(f"Loaded data size: {df.shape}")
    
    # Select and verify required columns


    required_columns = ['Search term', 'Product name', 'Level1 category']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required column: {missing_columns}")
        print("Available columns:")
        print(df.columns.tolist())
        return None
    
    # Filter only items where level 1 is 'Food' (performance optimization -select only required columns)


    food_df = df.loc[df['Level1 category'] == 'Food', required_columns].copy()
    print(f"Number of Food Items: {len(food_df)}")
    
    # NaN value handling


    food_df['Search term'] = food_df['Search term'].fillna('Direct access')
    food_df['Product name'] = food_df['Product name'].fillna('')
    
    # Remove duplicates (required)


    food_df_unique = food_df.drop_duplicates(subset=['Search term', 'Product name'])
    print(f"Number of items after deduplication: {len(food_df_unique)}")
    
    # Add result column


    if 'Level2 category' not in food_df_unique.columns:
        food_df_unique['Level2 category'] = None
    
    # Checkpoint file path


    timestamp = datetime.now().strftime('%y%m%d%h%m%s')
    checkpoint_file = f"level2 checkpoint{timestamp}.csv"
    
    # Check previous checkpoints (optimized: load once, apply once)


    import glob
    checkpoint_files = sorted(glob.glob("Level2 checkpoint *.csv"), reverse=True)
    
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[0]
        resume_prompt = input(f"Previous checkpoint ({latest_checkpoint}) was found. Would you like to proceed further? (y/n): ")
        
        if resume_prompt.lower() == 'Y':
            try:
                # Checkpoint efficient loading (only necessary columns)


                checkpoint_df = pd.read_csv(latest_checkpoint, usecols=['Search term', 'Product name', 'Level2 category'])
                
                # Efficient mapping (using merge)


                checkpoint_df = checkpoint_df.dropna(subset=['Level2 category'])
                
                # Generate merge key


                food_df_unique['Merge key'] = food_df_unique['Search term'] + '|' + food_df_unique['Product name']
                checkpoint_df['Merge key'] = checkpoint_df['Search term'] + '|' + checkpoint_df['Product name']
                
                # Create a mapping dictionary


                category_map = dict(zip(checkpoint_df['Merge key'], checkpoint_df['Level2 category']))
                
                # Apply quick mapping


                mask = food_df_unique['Merge key'].isin(category_map.keys())
                food_df_unique.loc[mask, 'Level2 category'] = food_df_unique.loc[mask, 'Merge key'].map(category_map)
                
                # Remove temporary column


                food_df_unique.drop('Merge key', axis=1, inplace=True)
                
                # Check number of items processed


                processed_count = food_df_unique['Level2 category'].notna().sum()
                print(f"At the checkpoint {processed_count} Item restored")
                
                # memory cleanup


                del checkpoint_df, category_map
                gc.collect()
            except Exception as e:
                print(f"Failed to load checkpoint: {str(e)}")
    
    # Identify items that have not yet been processed


    unprocessed_df = food_df_unique[food_df_unique['Level2 category'].isna()].copy()
    
    if len(unprocessed_df) == 0:
        print("All items have already been processed.")
        if output_path:
            food_df_unique.to_csv(output_path, index=False)
            print(f"The result is {output_path}saved.")
        return food_df_unique
    
    print(f"Number of items to process: {len(unprocessed_df)}")
    
    # Check CUDA availability (same as existing code)


    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA enabled: {device_name}")
        
        # Check VRAM


        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"vram: {total_memory:.2f} Gb")
        
        # Automatically adjust batch size (optional)


        if total_memory < 6:  # Less than 6GB


            batch_size = 1
            print("We have less VRAM, so we adjust the batch size to 1.")
        elif total_memory < 8:  # Less than 8GB (8GB for RTX 3070)


            batch_size = 2
            print("Adjust the batch size to 2.")
        elif total_memory < 12:  # Less than 12GB


            batch_size = 3
            print("Adjust the batch size to 3.")
    else:
        print("CUDA disabled: Run in CPU mode.")
        batch_size = 1
    
    # Chunk-wise processing (chunk_size units)


    total_items = len(unprocessed_df)
    total_chunks = (total_items + chunk_size - 1) // chunk_size
    
    # Total start time


    total_start_time = time.time()
    save_thread = None  # For tracking asynchronous storage threads


    
    try:
        for chunk_idx in range(total_chunks):
            # If asynchronous save is in progress, wait for completion


            if save_thread and save_thread.is_alive():
                print("Waiting for completion of saving previous checkpoint...")
                save_thread.join()
            
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, total_items)
            
            print(f"\nChunk {chunk_idx+1}/{total_chunks} Processing (item {chunk_start+1}-{chunk_end}/{total_items})")
            
            # Chunk data preparation


            chunk_indices = unprocessed_df.index[chunk_start:chunk_end]
            chunk_df = unprocessed_df.loc[chunk_indices]
            
            search_terms = chunk_df['Search term'].tolist()
            product_names = chunk_df['Product name'].tolist()
            
            # Chunk start time


            chunk_start_time = time.time()
            
            # Classification run


            chunk_results = classify_with_ollama(
                search_terms,
                product_names,
                batch_size=batch_size,
                use_cuda=cuda_available,
                model="Llama3:latest"  # Modify model name


            )
            
            # Results update


            for i, idx in enumerate(chunk_indices):
                food_df_unique.loc[idx, 'Level2 category'] = chunk_results[i]
            
            # Chunk processing time and estimated time


            chunk_time = time.time() - chunk_start_time
            items_per_second = len(chunk_df) / chunk_time if chunk_time > 0 else 0
            remaining_chunks = total_chunks - (chunk_idx + 1)
            estimated_time = remaining_chunks * chunk_time / 60  # minute by minute


            
            print(f"Chunk processing complete: {chunk_time:.1f}Takes seconds ({items_per_second:.2f} items/sec)")
            print(f"Estimated time remaining: {estimated_time:.1f}minute")
            
            # Save results to date (save checkpoints asynchronously)


            save_thread = save_checkpoint_async(food_df_unique, checkpoint_file)
            
            # Check category distribution


            current_results = food_df_unique[food_df_unique['Level2 category'].notna()]
            category_counts = current_results['Level2 category'].value_counts()
            
            print("\nCurrent category distribution:")
            for category, count in category_counts.items():
                percentage = count / len(current_results) * 100
                print(f"{category}: {count} ({percentage:.1f}%)")
            
            # memory cleanup


            gc.collect()
            if cuda_available:
                torch.cuda.empty_cache()
            
            # Rest between chunks (reduced from 30 seconds to 20 seconds)


            if chunk_idx < total_chunks - 1:
                print(f"20 seconds rest for the next chunk…")
                time.sleep(20)
        
        # Wait for final thread to complete


        if save_thread and save_thread.is_alive():
            print("Waiting for final checkpoint save to complete...")
            save_thread.join()
        
        # total time


        total_time = time.time() - total_start_time
        print(f"\nFull processing completed: {total_time/60:.1f}It takes minutes")
        
        # Check category distribution


        category_counts = food_df_unique['Level2 category'].value_counts()
        print("\nFinal category distribution:")
        for category, count in category_counts.items():
            percentage = count / len(food_df_unique) * 100
            print(f"{category}: {count} ({percentage:.1f}%)")
        
        # Save final result


        if output_path is None:
            output_path = f"food level2 classification{timestamp}.csv"
        
        # Optimize file size by storing only essential columns


        essential_cols = ['Search term', 'Product name', 'Level1 category', 'Level2 category']
        cols_to_save = [col for col in essential_cols if col in food_df_unique.columns]
        food_df_unique[cols_to_save].to_csv(output_path, index=False)
        print(f"Final result saved: {output_path}")
        
        return food_df_unique
    
    except KeyboardInterrupt:
        print("\nUser interruption. Saves the results up to now.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}. Saves the results up to now.")
    finally:
        # emergency storage


        emergency_file = f"emergency level2 save{timestamp}.csv"
        try:
            # Improve emergency storage speed by saving only essential columns


            essential_cols = ['Search term', 'Product name', 'Level1 category', 'Level2 category']
            cols_to_save = [col for col in essential_cols if col in food_df_unique.columns]
            food_df_unique[cols_to_save].to_csv(emergency_file, index=False)
            print(f"Emergency save completed: {emergency_file}")
        except:
            print("Emergency save failure")
    
    return food_df_unique

# main executable code


if __name__ == "Main":
    # GPU layer settings


    os.environ['Ollama gpu layers'] = '36'  # decreased from 43 to 36


    os.environ['Cuda visible devices'] = '0'
    
    # Data file path settings


    data_file = "All food items combinedv2.csv"
    print(f"Data files to classify: {data_file}")
    
    # Set batch size


    batch_size = int(input("Enter the batch size (default: 3, RTX 3070 recommended: 2-3): ") or "3")
    
    # Set checkpoint storage interval


    chunk_size = int(input("Set checkpoint saving interval (default: 500): ") or "500")
    
    # Process the entire dataset


    print("\nStart processing the entire dataset...")
    result_df = process_food_classification(
        data_path=data_file,
        batch_size=batch_size,
        chunk_size=chunk_size
    )
    
    print("Program exit")