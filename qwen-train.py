# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import re # Import regex for parsing

# Constants for chat format
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ASSISTANT_ROLE = "assistant"
MAX_LENGTH = 512  # Max sequence length

# Step 1: Load the Qwen3 1.7B model and tokenizer
model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token_id is None: # Ensure pad_token_id is set if pad_token was set to eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


# Step 2: Load the dataset
dataset = load_dataset("json", data_files="tool_training_dataset.jsonl", split="train")

# Split into training and validation sets
if len(dataset) > 1:
    dataset_splits = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset_splits["train"]
    val_dataset = dataset_splits["test"]
else:
    train_dataset = dataset
    val_dataset = dataset.select(range(min(1, len(dataset)))) # Use 1 sample for val if dataset has 1, else empty


# Step 3: Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Step 4: Tokenize the dataset and prepare labels
def tokenize_and_prepare_labels(examples):
    tokenized_batch = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_offsets_mapping=True
    )

    labels_batch = []
    for i in range(len(examples["text"])):
        raw_text = examples["text"][i]
        input_ids_single = tokenized_batch.input_ids[i]
        offset_mapping_single = tokenized_batch.offset_mapping[i]
        
        current_labels = [-100] * len(input_ids_single)
        
        assistant_predict_char_spans = []
        cursor = 0
        while cursor < len(raw_text):
            marker = f"{IM_START}{ASSISTANT_ROLE}\n"
            start_marker_idx = raw_text.find(marker, cursor)
            if start_marker_idx == -1:
                break
            
            predict_char_start = start_marker_idx + len(marker)
            
            # Find the end of this assistant's utterance, including its <|im_end|>
            # This assumes <|im_end|> is not part of assistant's actual content before its own delimiter
            # A more robust parser might be needed for deeply nested or complex structures,
            # but for typical Qwen chat format, this should be okay.
            
            # We need to find the <|im_end|> that corresponds to THIS <|im_start|>assistant
            # This can be tricky. A simple find might get an earlier <|im_end|> if roles are nested.
            # However, Qwen format is usually flat: <|im_start|>role1...<|im_end|><|im_start|>role2...<|im_end|>
            
            # Let's find the next <|im_start|> to delimit the current assistant block, or end of string
            next_im_start_after_assistant_content_start = raw_text.find(IM_START, predict_char_start)
            
            current_assistant_block_text = ""
            if next_im_start_after_assistant_content_start != -1:
                current_assistant_block_text = raw_text[predict_char_start:next_im_start_after_assistant_content_start]
            else: # This assistant block goes to the end of the raw_text
                current_assistant_block_text = raw_text[predict_char_start:]

            # The actual content to predict must end with this assistant's <|im_end|>
            # Find the last <|im_end|> within this current_assistant_block_text
            # This assumes the assistant's own <|im_end|> is the one we want.
            idx_of_assistants_im_end = current_assistant_block_text.rfind(IM_END)

            if idx_of_assistants_im_end != -1:
                predict_char_end = predict_char_start + idx_of_assistants_im_end + len(IM_END)
                assistant_predict_char_spans.append((predict_char_start, predict_char_end))
                cursor = predict_char_end
            else:
                # This assistant block seems not to have its own <|im_end|> before another <|im_start|> or end of text.
                # This could be an issue with data generation or format.
                # For safety, advance cursor past the marker to avoid infinite loop.
                cursor = predict_char_start
                if cursor == start_marker_idx + len(marker): # if cursor didn't advance
                    cursor +=1 # force advance

        for token_idx, (offset_start, offset_end) in enumerate(offset_mapping_single):
            if input_ids_single[token_idx] == tokenizer.pad_token_id:
                current_labels[token_idx] = -100
                continue
            
            if offset_end == 0: # Special token added by tokenizer, not from original string
                continue

            for span_s, span_e in assistant_predict_char_spans:
                # Check if the token's character span (offset_start, offset_end)
                # overlaps with the prediction span (span_s, span_e)
                if max(offset_start, span_s) < min(offset_end, span_e):
                    current_labels[token_idx] = input_ids_single[token_idx]
                    break
        labels_batch.append(current_labels)
        
    tokenized_batch["labels"] = labels_batch
    return tokenized_batch

processed_train_dataset = train_dataset.map(tokenize_and_prepare_labels, batched=True, remove_columns=["text"])
processed_val_dataset = val_dataset.map(tokenize_and_prepare_labels, batched=True, remove_columns=["text"])


# Step 5: Set up training arguments
training_args = TrainingArguments(
 output_dir="./results", # Directory to save results
 eval_strategy="steps", # Evaluate at the end of each epoch (now using steps)
 eval_steps=75, # Number of steps per epoch (300 examples / 4 batch_size)
 learning_rate=1e-4, # Learning rate
 per_device_train_batch_size=4, # Batch size for training
 per_device_eval_batch_size=4, # Batch size for evaluation
 num_train_epochs=3, # Number of training epochs
 weight_decay=0.01, # Weight decay for regularization
 logging_dir="./logs", # Directory for logs
 logging_steps=10, # Log every 10 steps
 save_strategy="steps", # Save model at each epoch (now using steps)
 save_steps=75, # Number of steps per epoch
 load_best_model_at_end=True, # Load the best model based on evaluation
 fp16=True # Use mixed precision for faster training
)

# Step 6: Initialize the Trainer
trainer = Trainer(
 model=model,
 args=training_args,
 train_dataset=processed_train_dataset,
 eval_dataset=processed_val_dataset,
 tokenizer=tokenizer # Tokenizer is still useful for saving, and for DataCollator if used
)

# Step 7: Train the model
trainer.train()

# Step 8: Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_qwen3_lora")
tokenizer.save_pretrained("./fine_tuned_qwen3_lora")
