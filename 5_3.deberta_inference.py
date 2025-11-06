import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DebertaV2Config
from tqdm import tqdm
import logging
import glob
import gc
from datetime import datetime

# Logging settings

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/Inference{datetime.now().strftime('%y%m%d%h%m%s')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# setting

INPUT_DIR = '/Mnt/c/users/airja/01 chat/tab data/tab data/one drive 4 3 17 2025/deberta/preprocessed data'  # actual data path

OUTPUT_DIR = './predictions'  # Directory to store prediction results

SAMPLE_DIR = './samples'  # Directory to store sample results

MODEL_PATH = './models/best model step 14000'  # Best performing model path

BATCH_SIZE = 32  # Batch size optimized for RTX 3070

MAX_LEN = 256    # Token length same as training

CATEGORIES_PATH = '/Mnt/c/users/airja/01 chat/tab data/tab data/one drive 4 3 17 2025/deberta/preprocessed data/predefined categories.json'  # Category Information File

INDICES_PATH = '/Mnt/c/users/airja/01 chat/tab data/tab data/one drive 4 3 17 2025/deberta/preprocessed data/category indices.json'  # Category Index File

FP16 = True      # Use mixed precision

SAMPLE_SIZE = 1000  # sample size


# Create output directory

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# GPU settings

device = torch.device('Cuda' if torch.cuda.is_available() else 'Cpu')
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"gpu: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")

# Load category information

def load_category_info():
    logger.info("Loading category information...")
    with open(CATEGORIES_PATH, 'R') as f:
        categories = json.load(f)
    
    with open(INDICES_PATH, 'R') as f:
        indices = json.load(f)
    
    # Create index-category mapping (reverse mapping)

    idx_to_category = {}
    
    # Level 1 processing

    idx_to_category['Level1'] = {v: k for k, v in indices['Level1'].items()}
    
    # Level 2 processing

    idx_to_category['Level2'] = {}
    for parent, children in indices['Level2'].items():
        for child, idx in children.items():
            idx_to_category['Level2'][idx] = child
    
    # Level 3 processing

    idx_to_category['To fluff'] = {}
    for parent, children in indices['To fluff'].items():
        for child, idx in children.items():
            idx_to_category['To fluff'][idx] = child
    
    return categories, indices, idx_to_category

# Model classes for hierarchical classification

class HierarchicalDebertaClassifier(torch.nn.Module):
    def __init__(self, model_path, num_level1, num_level2, num_level3):
        super(HierarchicalDebertaClassifier, self).__init__()
        self.num_level1 = num_level1
        self.num_level2 = num_level2
        self.num_level3 = num_level3
        
        # Setting up and loading DeBERTa

        config = DebertaV2Config.from_pretrained(model_path)
        config.output_hidden_states = True
        
        # load model

        logger.info("Loading DeBERTa model...")
        self.deberta = DebertaV2ForSequenceClassification.from_pretrained(
            model_path,
            config=config
        )
        
        # Additional sorting head

        hidden_size = self.deberta.config.hidden_size
        self.dropout = torch.nn.Dropout(0.2)
        self.level2_classifier = torch.nn.Linear(hidden_size, self.num_level2)
        self.level3_classifier = torch.nn.Linear(hidden_size, self.num_level3)
    
    def forward(self, input_ids, attention_mask):
        # Obtaining the output of DeBERTa

        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Level 1 classification (Food/Non-Food)

        level1_logits = outputs.logits
        
        # Get hidden state

        hidden_states = outputs.hidden_states[-1][:, 0, :]
        hidden_states = self.dropout(hidden_states)
        
        # Level 2 and 3 classification

        level2_logits = self.level2_classifier(hidden_states)
        level3_logits = self.level3_classifier(hidden_states)
        
        return level1_logits, level2_logits, level3_logits

# Dataset class

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        
        # Tokenizing

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='Max length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='Pt'
        )
        
        return {
            'Input ids': encoding['Input ids'].flatten(),
            'Attention mask': encoding['Attention mask'].flatten(),
            'Text': text
        }

# Inference function -modified part

def predict_batch(model, dataloader, device, idx_to_category, fp16=False):
    model.eval()
    
    # Find the index corresponding to the Food category

    food_idx = None
    for idx, category in idx_to_category['Level1'].items():
        if category == 'Food':
            food_idx = idx
            break
    
    if food_idx is None:
        logger.error("Could not find 'Food' category in idx_to_category mapping")
        return []
    
    logger.info(f"Food category index: {food_idx}")
    
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Go to data device

            input_ids = batch['Input ids'].to(device)
            attention_mask = batch['Attention mask'].to(device)
            texts = batch['Text']
            
            # prediction

            if fp16:
                with torch.cuda.amp.autocast():
                    level1_logits, level2_logits, level3_logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
            else:
                level1_logits, level2_logits, level3_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Forecast transformation

            level1_preds = torch.argmax(level1_logits, dim=1).cpu().numpy()
            level2_preds = torch.argmax(level2_logits, dim=1).cpu().numpy()
            level3_preds = torch.argmax(level3_logits, dim=1).cpu().numpy()
            
            # Food Item Identification -Use Correct Index

            is_food = (level1_preds == food_idx)
            
            # Save results

            for i in range(len(texts)):
                level1_category = idx_to_category['Level1'][level1_preds[i]]
                
                # Use level 2 and level 3 prediction only in the case of food

                if is_food[i]:
                    level2_category = idx_to_category['Level2'][level2_preds[i]]
                    level3_category = idx_to_category['To fluff'][level3_preds[i]]
                else:
                    level2_category = "N/a"
                    level3_category = "N/a"
                
                predictions.append({
                    'Text': texts[i],
                    'Level1 category': level1_category,
                    'Level2 category': level2_category,
                    'Level3 category': level3_category,
                    'Level1 idx': int(level1_preds[i]),
                    'Level2 idx': int(level2_preds[i]) if is_food[i] else -1,
                    'Level3 idx': int(level3_preds[i]) if is_food[i] else -1
                })
            
            # memory cleanup

            del input_ids, attention_mask
            del level1_logits, level2_logits, level3_logits
    
    return predictions

# Sample processing and preview functions

def process_sample(df, model, tokenizer, device, idx_to_category, text_column, file_name):
    logger.info(f"processing {SAMPLE_SIZE} sample rows for preview")
    
    # sample extraction

    sample_df = df.head(SAMPLE_SIZE).copy().reset_index(drop=True)
    texts = sample_df[text_column].fillna("").astype(str).tolist()
    
    # Create dataset and data loader

    dataset = InferenceDataset(texts, tokenizer, MAX_LEN)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # perform predictions

    predictions = predict_batch(model, dataloader, device, idx_to_category, FP16)
    
    # Convert result to DataFrame

    results_df = pd.DataFrame(predictions)
    
    # Merge results with original dataframe

    sample_with_predictions = pd.concat([sample_df, 
                                         results_df[['Level1 category', 'Level2 category', 'Level3 category', 
                                                   'Level1 idx', 'Level2 idx', 'Level3 idx']]], axis=1)
    
    # Save sample results

    sample_output_file = os.path.join(SAMPLE_DIR, f"sample{file_name}")
    sample_with_predictions.to_csv(sample_output_file, index=False)
    logger.info(f"Saved sample predictions to {sample_output_file}")
    
    # Statistical output

    food_count = (results_df['Level1 category'] == 'Food').sum()
    level2_counts = results_df[results_df['Level1 category'] == 'Food']['Level2 category'].value_counts().head(5)
    
    logger.info(f"Sample Statistics:")
    logger.info(f"-Total Samples: {len(results_df)}")
    logger.info(f"-Food Items: {food_count} ({food_count/len(results_df)*100:.2f}%)")
    logger.info(f"-Top 5 Level 2 Categories:")
    for cat, count in level2_counts.items():
        logger.info(f"  * {cat}: {count} ({count/food_count*100:.2f}% of Food items)")
    
    # Print sample results to console

    print("\n==== SAMPLE PREDICTION RESULTS ====")
    print(f"file: {file_name}")
    print(f"Total Samples: {len(results_df)}")
    print(f"Food Items: {food_count} ({food_count/len(results_df)*100:.2f}%)")
    print("\nSample Rows (first 10):")
    
    # Select column to print

    display_cols = [text_column, 'Level1 category', 'Level2 category', 'Level3 category']
    
    # Function for adjusting column width

    def truncate_str(s, max_len=40):
        if isinstance(s, str) and len(s) > max_len:
            return s[:max_len-3] + "..."
        return s
    
    # Print first 10 rows

    for i, row in sample_with_predictions.head(10).iterrows():
        print("-" * 100)
        for col in display_cols:
            value = truncate_str(row[col])
            print(f"{col.upper()}: {value}")
    
    print("=" * 100)
    print(f"entire {SAMPLE_SIZE}dog sample results {sample_output_file}saved.")
    print("=" * 100)
    
    return sample_with_predictions

# main inference function

def main():
    start_time = datetime.now()
    logger.info(f"Starting inference process...")
    
    try:
        # Load category information

        categories, indices, idx_to_category = load_category_info()
        
        # Calculate number of categories per level

        num_level1 = len(categories['Level1'])
        num_level2 = len(categories['Level2']['Food'])
        num_level3 = max(len(children) for children in categories['To fluff'].values())
        
        logger.info(f"Number of categories -Level 1: {num_level1}, Level 2: {num_level2}, Level 3: {num_level3}")
        
        # tokenizer load

        logger.info(f"Loading tokenizer from {MODEL_PATH}")
        tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)
        
        # Model initialization

        logger.info(f"Initializing model from {MODEL_PATH}")
        model = HierarchicalDebertaClassifier(MODEL_PATH, num_level1, num_level2, num_level3)
        model.to(device)
        
        # Get file list -what's fixed

        csv_files = glob.glob(os.path.join(INPUT_DIR, "*empty*.csv"))
        logger.info(f"found {len(csv_files)} CSV files to process")
        
        # Processing by file

        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            logger.info(f"Processing file: {file_name}")
            
            # Load CSV file

            try:
                df = pd.read_csv(file_path)
                logger.info(f"Loaded data shape: {df.shape}")
                
                # Check text column name (only done once)

                text_column = None
                for col in ['Text', 'Content', 'Description', 'Product name', 'Search term']:
                    if col in df.columns:
                        text_column = col
                        break
                
                if text_column is None:
                    logger.error(f"Could not find text column in {file_name}")
                    continue
                
                logger.info(f"Using text column: {text_column}")
                
                # Sample processing and preview

                sample_results = process_sample(df, model, tokenizer, device, idx_to_category, text_column, file_name)
                
                # Prompt user to proceed

                user_input = input("\nDo you want to continue processing the entire file? (y/n): ")
                if user_input.lower() != 'Y':
                    logger.info(f"Skipping full processing of {file_name} as per user request")
                    continue
                
                # Text data preparation

                texts = df[text_column].fillna("").astype(str).tolist()
                
                # Create dataset and data loader

                dataset = InferenceDataset(texts, tokenizer, MAX_LEN)
                dataloader = DataLoader(
                    dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True
                )
                
                logger.info(f"Number of samples: {len(dataset)}")
                logger.info(f"Number of batches: {len(dataloader)}")
                
                # GPU memory status logging

                if torch.cuda.is_available():
                    logger.info(f"GPU memory before inference: {torch.cuda.memory_allocated(0) / 1e9:.2f} Gb")
                
                # perform predictions

                predictions = predict_batch(model, dataloader, device, idx_to_category, FP16)
                
                # Convert result to DataFrame

                results_df = pd.DataFrame(predictions)
                
                # Merge results with original dataframe

                df = df.reset_index(drop=True)  # Add index reset

                df_with_predictions = pd.concat([df, results_df[['Level1 category', 'Level2 category', 'Level3 category', 
                                                             'Level1 idx', 'Level2 idx', 'Level3 idx']]], axis=1)
                
                # Save results

                output_file = os.path.join(OUTPUT_DIR, f"before{file_name}")
                df_with_predictions.to_csv(output_file, index=False)
                logger.info(f"Saved predictions to {output_file}")
                
                # memory cleanup

                del df, results_df, df_with_predictions, dataset, dataloader, predictions
                gc.collect()
                torch.cuda.empty_cache()
                
                if torch.cuda.is_available():
                    logger.info(f"GPU memory after cleanup: {torch.cuda.memory_allocated(0) / 1e9:.2f} Gb")
                
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {str(e)}", exc_info=True)
        
        # total processing time

        end_time = datetime.now()
        processing_duration = end_time - start_time
        logger.info(f"Total processing time: {processing_duration}")
        
    except Exception as e:
        logger.error(f"Critical error: {str(e)}", exc_info=True)
        raise

if __name__ == "Main":
    main()