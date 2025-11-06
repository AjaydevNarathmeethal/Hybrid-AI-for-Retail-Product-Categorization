import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DebertaV2Config
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import logging
import gc
import time
from datetime import datetime

# Record execution start time

start_time = datetime.now()

# Logging settings

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/Training{start_time.strftime('%y%m%d%h%m%s')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Settings -RTX 3070 Optimization

INPUT_DIR = '/Home/airjaehyuk/preprocessed data'
OUTPUT_DIR = './models'
CHECKPOINT_DIR = "./checkpoints"
MODEL_NAME = 'Microsoft/deberta vz base'
BATCH_SIZE = 16  # increased batch size

EPOCHS = 2       # Change the number of epochs to 2

LEARNING_RATE = 5e-5  # increased learning rate

MAX_LEN = 256    # Decrease token length

EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 2  # Reduced gradient accumulation step

WARMUP_RATIO = 0.1  # Rate-based warm-up

SAVE_STEPS = 1000
EVAL_STEPS = 1000
FP16 = True      # Use mixed precision

TRAIN_VAL_SPLIT = 0.9
LOG_INTERVAL = 20  # log output interval


# Create directory

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# GPU settings

device = torch.device('Cuda' if torch.cuda.is_available() else 'Cpu')
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"gpu: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")

# Save settings

config = {
    'Model name': MODEL_NAME,
    'Batch size': BATCH_SIZE,
    'Gradient accumulation steps': GRADIENT_ACCUMULATION_STEPS,
    'Effective batch size': BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
    'Epochs': EPOCHS,
    'Learning rate': LEARNING_RATE,
    'Max len': MAX_LEN,
    'Fp16': FP16,
}
logger.info(f"configuration: {config}")

# Load category information

def load_category_info():
    logger.info("Loading category information...")
    with open(os.path.join(INPUT_DIR, 'Predefined categories.json'), 'R') as f:
        categories = json.load(f)
    
    with open(os.path.join(INPUT_DIR, 'Category indices.json'), 'R') as f:
        indices = json.load(f)
    
    return categories, indices

# Dataset Class -Improved Efficiency

class HierarchicalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Extract text columns in advance and save them as a list (memory efficiency)

        self.texts = dataframe['Text'].values.tolist()
        
        # Pre-extract labels (if any)

        if 'Level1 label' in dataframe.columns:
            self.has_labels = True
            self.level1_labels = dataframe['Level1 label'].values
            self.level2_labels = dataframe['Level2 label'].values
            self.level3_labels = dataframe['Level3 label'].values
            self.is_food = (dataframe['Level1 category'] == 'Food').astype(int).values
        else:
            self.has_labels = False
    
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
        
        item = {
            'Input ids': encoding['Input ids'].flatten(),
            'Attention mask': encoding['Attention mask'].flatten(),
        }
        
        # Add labels only for training data

        if self.has_labels:
            item['Level1 label'] = torch.tensor(self.level1_labels[index], dtype=torch.long)
            item['Level2 label'] = torch.tensor(self.level2_labels[index] if self.level2_labels[index] >= 0 else 0, dtype=torch.long)
            item['Level3 label'] = torch.tensor(self.level3_labels[index] if self.level3_labels[index] >= 0 else 0, dtype=torch.long)
            item['Is food'] = torch.tensor(self.is_food[index], dtype=torch.long)
        
        return item

# Model classes for hierarchical classification

class HierarchicalDebertaClassifier(torch.nn.Module):
    def __init__(self, model_name, num_level1, num_level2, num_level3):
        super(HierarchicalDebertaClassifier, self).__init__()
        self.num_level1 = num_level1
        self.num_level2 = num_level2
        self.num_level3 = num_level3
        
        # Setting up and loading DeBERTa -making memory efficient

        config = DebertaV2Config.from_pretrained(model_name)
        config.num_labels = self.num_level1
        config.output_hidden_states = True
        config.hidden_dropout_prob = 0.2  # Increased dropouts

        config.attention_probs_dropout_prob = 0.2  # Increased dropouts

        
        # Model Loading -Heavy Parts

        logger.info("Loading DeBERTa model...")
        self.deberta = DebertaV2ForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
        
        # Additional sorting head

        hidden_size = self.deberta.config.hidden_size
        self.dropout = torch.nn.Dropout(0.2)  # Additional dropouts

        self.level2_classifier = torch.nn.Linear(hidden_size, self.num_level2)
        self.level3_classifier = torch.nn.Linear(hidden_size, self.num_level3)
        
        # weight initialization

        torch.nn.init.xavier_normal_(self.level2_classifier.weight)
        torch.nn.init.xavier_normal_(self.level3_classifier.weight)
    
    def forward(self, input_ids, attention_mask, is_food=None):
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
        hidden_states = self.dropout(hidden_states)  # Apply dropout

        
        # Level 2, 3 classification (only for food items)

        level2_logits = self.level2_classifier(hidden_states)
        level3_logits = self.level3_classifier(hidden_states)
        
        return level1_logits, level2_logits, level3_logits

# Loss function (for hierarchical classification)

class HierarchicalLoss(torch.nn.Module):
    def __init__(self, level1_weight=0.6, level2_weight=0.2, level3_weight=0.2):
        super(HierarchicalLoss, self).__init__()
        self.level1_weight = level1_weight
        self.level2_weight = level2_weight
        self.level3_weight = level3_weight
        self.ce_loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, level1_logits, level2_logits, level3_logits, 
               level1_labels, level2_labels, level3_labels, is_food):
        # Level 1 Loss Calculation

        level1_loss = self.ce_loss(level1_logits, level1_labels)
        
        # Level 2, 3 loss calculation (for Food items only)

        if is_food.sum() > 0:
            level2_loss = self.ce_loss(level2_logits[is_food == 1], level2_labels[is_food == 1])
            level3_loss = self.ce_loss(level3_logits[is_food == 1], level3_labels[is_food == 1])
        else:
            # Set to 0 if there is no Food item

            level2_loss = torch.tensor(0.0, device=level1_logits.device)
            level3_loss = torch.tensor(0.0, device=level1_logits.device)
        
        # Calculate total loss as a weighted sum

        total_loss = (self.level1_weight * level1_loss + 
                     self.level2_weight * level2_loss + 
                     self.level3_weight * level3_loss)
        
        return total_loss, level1_loss, level2_loss, level3_loss

# Evaluation function -a more efficient version

def evaluate(model, dataloader, device, amp_scaler=None):
    model.eval()
    
    level1_preds = []
    level1_labels = []
    food_items = []
    level2_preds_food = []
    level2_labels_food = []
    level3_preds_food = []
    level3_labels_food = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Go to data device

            input_ids = batch['Input ids'].to(device)
            attention_mask = batch['Attention mask'].to(device)
            level1_label = batch['Level1 label'].to(device)
            level2_label = batch['Level2 label'].to(device)
            level3_label = batch['Level3 label'].to(device)
            is_food = batch['Is food'].to(device)
            
            # prediction

            if amp_scaler is not None:
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
            
            # Collecting forecasts -memory efficient way

            level1_pred = torch.argmax(level1_logits, dim=1).cpu().numpy()
            level1_preds.extend(level1_pred)
            level1_labels.extend(level1_label.cpu().numpy())
            
            # Food Item Identification

            is_food_np = is_food.cpu().numpy()
            food_items.extend(is_food_np)
            
            # Store level 2 and 3 predictions only for Food items

            food_indices = np.where(is_food_np == 1)[0]
            if len(food_indices) > 0:
                level2_preds_food.extend(torch.argmax(level2_logits[is_food == 1], dim=1).cpu().numpy())
                level2_labels_food.extend(level2_label[is_food == 1].cpu().numpy())
                level3_preds_food.extend(torch.argmax(level3_logits[is_food == 1], dim=1).cpu().numpy())
                level3_labels_food.extend(level3_label[is_food == 1].cpu().numpy())
            
            # Clean up memory after batch processing

            del input_ids, attention_mask, level1_label, level2_label, level3_label, is_food
            del level1_logits, level2_logits, level3_logits
    
    # Level 1 Metric Calculations

    level1_accuracy = accuracy_score(level1_labels, level1_preds)
    level1_f1 = f1_score(level1_labels, level1_preds, average='Weighted')
    
    # Calculate level 2 and 3 metrics for Food items

    if len(level2_preds_food) > 0:
        level2_accuracy = accuracy_score(level2_labels_food, level2_preds_food)
        level2_f1 = f1_score(level2_labels_food, level2_preds_food, average='Weighted')
        level3_accuracy = accuracy_score(level3_labels_food, level3_preds_food)
        level3_f1 = f1_score(level3_labels_food, level3_preds_food, average='Weighted')
    else:
        level2_accuracy = level2_f1 = level3_accuracy = level3_f1 = 0
    
    # memory cleanup

    torch.cuda.empty_cache()
    
    results = {
        'Level1 accuracy': float(level1_accuracy),
        'Level1 f1': float(level1_f1),
        'Level2 accuracy': float(level2_accuracy),
        'Level2 f1': float(level2_f1),
        'Level3 accuracy': float(level3_accuracy),
        'Level3 f1': float(level3_f1)
    }
    
    return results

# main function

def main():
    logger.info(f"Starting training process...")
    
    # Load category information

    categories, indices = load_category_info()
    
    # Calculate number of categories per level

    num_level1 = len(categories['Level1'])
    num_level2 = len(categories['Level2']['Food'])
    
    # The number of level 3 categories is the maximum of the number of level 3 categories across all level 2 categories.

    num_level3 = max(len(children) for children in categories['To fluff'].values())
    
    logger.info(f"Number of categories -Level 1: {num_level1}, Level 2: {num_level2}, Level 3: {num_level3}")
    
    # tokenizer load

    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    
    # data file path

    train_file = os.path.join(INPUT_DIR, 'Preprocessed train.csv')
    logger.info(f"Loading training data from {train_file}")
    
    # Loading and Splitting Data

    try:
        df = pd.read_csv(train_file)
        logger.info(f"Loaded data shape: {df.shape}")
        
        # Training/validation split

        np.random.seed(42)  # Seed settings for reproducibility

        train_size = int(TRAIN_VAL_SPLIT * len(df))
        indices = np.random.permutation(len(df))
        train_indices, val_indices = indices[:train_size], indices[train_size:]
        
        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)
        
        logger.info(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}")
        
        # Create dataset

        train_dataset = HierarchicalDataset(train_df, tokenizer, MAX_LEN)
        val_dataset = HierarchicalDataset(val_df, tokenizer, MAX_LEN)
        
        # Create data loader

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True  # Drop last batch (memory efficiency)

        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"Number of training batches: {len(train_dataloader)}")
        logger.info(f"Number of validation batches: {len(val_dataloader)}")
        
        # Model initialization

        logger.info(f"Initializing model: {MODEL_NAME}")
        model = HierarchicalDebertaClassifier(MODEL_NAME, num_level1, num_level2, num_level3)
        model.to(device)
        
        # Optimizer and scheduler settings

        no_decay = ['Bias', 'Layer norm.weight']
        optimizer_grouped_parameters = [
            {'Params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'Weight decay': 0.01},
            {'Params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'Weight decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
        
        # Step Count

        total_steps = len(train_dataloader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
        warmup_steps = int(WARMUP_RATIO * total_steps)
        
        logger.info(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # loss function

        criterion = HierarchicalLoss(level1_weight=0.6, level2_weight=0.2, level3_weight=0.2)
        
        # Mixed Precision Settings

        scaler = None
        if FP16 and torch.cuda.is_available():
            logger.info("Using mixed precision training (FP16)")
            scaler = torch.cuda.amp.GradScaler()
        
        # Start learning

        logger.info("Starting training...")
        model.train()
        
        # Peak Performance Tracking

        best_f1 = 0.0
        global_step = 0
        
        for epoch in range(EPOCHS):
            epoch_start_time = time.time()
            
            train_loss = 0.0
            train_steps = 0
            
            # progress bar

            progress_bar = tqdm(train_dataloader, desc=f"epoch {epoch+1}/{EPOCHS}")
            
            for step, batch in enumerate(progress_bar):
                # Periodic logging of memory information

                if step % 500 == 0 and torch.cuda.is_available():
                    logger.info(f"step {step} -GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} Gb")
                
                # Go to data device

                input_ids = batch['Input ids'].to(device)
                attention_mask = batch['Attention mask'].to(device)
                level1_label = batch['Level1 label'].to(device)
                level2_label = batch['Level2 label'].to(device)
                level3_label = batch['Level3 label'].to(device)
                is_food = batch['Is food'].to(device)
                
                # When using FP16

                if scaler is not None:
                    # forward spread

                    with torch.cuda.amp.autocast():
                        level1_logits, level2_logits, level3_logits = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        
                        # loss calculation

                        loss, l1_loss, l2_loss, l3_loss = criterion(
                            level1_logits, level2_logits, level3_logits,
                            level1_label, level2_label, level3_label, is_food
                        )
                        
                        # Scaling for gradient accumulation

                        loss = loss / GRADIENT_ACCUMULATION_STEPS
                    
                    # Backpropagation

                    scaler.scale(loss).backward()
                    
                    # Optimizer step after gradient accumulation

                    if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        # gradient clipping

                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        # Optimizer and scheduler steps

                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        global_step += 1
                        train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                        train_steps += 1
                
                # When using FP32

                else:
                    # forward spread

                    level1_logits, level2_logits, level3_logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    # loss calculation

                    loss, l1_loss, l2_loss, l3_loss = criterion(
                        level1_logits, level2_logits, level3_logits,
                        level1_label, level2_label, level3_label, is_food
                    )
                    
                    # Scaling for gradient accumulation

                    loss = loss / GRADIENT_ACCUMULATION_STEPS
                    
                    # Backpropagation

                    loss.backward()
                    
                    # Optimizer step after gradient accumulation

                    if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        # gradient clipping

                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        # Optimizer and scheduler steps

                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        global_step += 1
                        train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                        train_steps += 1
                
                # Progress bar updates

                if step % LOG_INTERVAL == 0:
                    progress_bar.set_postfix({
                        'Loss': f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}",
                        'L1 loss': f"{l1_loss.item():.4f}",
                        'Lr': f"{optimizer.param_groups[0]['Lr']:.1e}"
                    })
                
                # Periodic evaluation and checkpoint storage

                if global_step > 0 and global_step % EVAL_STEPS == 0:
                    # evaluation

                    logger.info(f"Evaluating at step {global_step}...")
                    eval_results = evaluate(model, val_dataloader, device, scaler)
                    logger.info(f"Evaluation results: {eval_results}")
                    
                    # Save the best performing model

                    if eval_results['Level1 f1'] > best_f1:
                        best_f1 = eval_results['Level1 f1']
                        
                        # Save model

                        best_model_path = os.path.join(OUTPUT_DIR, f'best model step{global_step}')
                        os.makedirs(best_model_path, exist_ok=True)
                        
                        model_to_save = model.module if hasattr(model, 'Module') else model
                        model_to_save.deberta.save_pretrained(best_model_path)
                        tokenizer.save_pretrained(best_model_path)
                        
                        # Save model information

                        with open(os.path.join(best_model_path, 'Model info.json'), 'W') as f:
                            json.dump({
                                'Step': global_step,
                                'Epoch': epoch + 1,
                                'Metrics': eval_results,
                                'Config': config
                            }, f, indent=2)
                        
                        logger.info(f"New best model saved with level1_f1: {best_f1:.4f}")
                    
                    # Switch back to training mode

                    model.train()
                
                # memory cleanup

                del input_ids, attention_mask, level1_label, level2_label, level3_label, is_food
                del level1_logits, level2_logits, level3_logits, loss
                
                # Clean up memory every 50 batches

                if step % 50 == 0:
                    torch.cuda.empty_cache()
            
            # Epoch end time

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            
            # Epoch average loss

            avg_train_loss = train_loss / train_steps if train_steps > 0 else 0
            
            logger.info(f"epoch {epoch+1}/{EPOCHS} completed in {epoch_duration:.2f} Seconds")
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # Evaluation after epoch

            logger.info(f"Evaluating after epoch {epoch+1}...")
            eval_results = evaluate(model, val_dataloader, device, scaler)
            logger.info(f"Evaluation results after epoch {epoch+1}: {eval_results}")
            
            # Save best performing model after epoch

            if eval_results['Level1 f1'] > best_f1:
                best_f1 = eval_results['Level1 f1']
                
                # Save model

                best_model_path = os.path.join(OUTPUT_DIR, f'best model epoch{epoch+1}')
                os.makedirs(best_model_path, exist_ok=True)
                
                model_to_save = model.module if hasattr(model, 'Module') else model
                model_to_save.deberta.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)
                
                # Save model information

                with open(os.path.join(best_model_path, 'Model info.json'), 'W') as f:
                    json.dump({
                        'Epoch': epoch + 1,
                        'Global step': global_step,
                        'Metrics': eval_results,
                        'Config': config
                    }, f, indent=2)
                
                logger.info(f"New best model saved with level1_f1: {best_f1:.4f}")
            
            # memory cleanup

            gc.collect()
            torch.cuda.empty_cache()
        
        # Save final model

        final_model_path = os.path.join(OUTPUT_DIR, 'Final model')
        os.makedirs(final_model_path, exist_ok=True)
        
        model_to_save = model.module if hasattr(model, 'Module') else model
        model_to_save.deberta.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        # Save model information

        with open(os.path.join(final_model_path, 'Model info.json'), 'W') as f:
            json.dump({
                'Final epoch': EPOCHS,
                'Final step': global_step,
                'Best f1': best_f1,
                'Final metrics': eval_results,
                'Config': config
            }, f, indent=2)
        
        logger.info(f"Final model saved to {final_model_path}")
        
        # total training time

        end_time = datetime.now()
        training_duration = end_time - start_time
        logger.info(f"Total training time: {training_duration}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "Main":
    try:
        main()
    except Exception as e:
        logger.error(f"Critical error: {str(e)}", exc_info=True)