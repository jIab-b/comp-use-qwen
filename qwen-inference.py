import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys # Import sys for command-line arguments
import json # For parsing model's JSON output
import subprocess # For calling the tool executor script
import re # For extracting JSON from model output

# Configuration
BASE_MODEL_NAME = "Qwen/Qwen3-0.6B"
LORA_ADAPTER_PATH = "./fine_tuned_qwen3_lora" # Path where LoRA adapter and its tokenizer were saved
TOOL_EXECUTOR_SCRIPT = "qwen_tool_executor.py"
SYSTEM_PROMPT_CONTENT = (
    "You are a helpful AI assistant. You are capable of file system operations "
    "(navigate, list, move, copy, delete) and running applications by calling tools. "
    "When you decide to use a tool, first think about the steps and then call the tool "
    "using the specified JSON format."
)
# Special tokens for conversation structure
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def extract_json_from_response(text_response):
    """
    Extracts the first valid JSON object from the model's text response.
    Looks for content between <think>...</think> and the JSON.
    """
    # Regex to find {"function_call": ...}
    # This regex tries to find a valid JSON object that starts with {"function_call"
    # It's a bit naive and might need improvement for more complex JSON.
    match = re.search(r'(\{[\s\S]*\"function_call\"[\s\S]*\})\s*(<\|im_end\|>)?$', text_response)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON: {json_str}")
            return None
    return None

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} \"<your user query>\"")
        sys.exit(1)
    
    user_query = sys.argv[1]

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the tokenizer from the LoRA adapter path
    # This ensures we use the exact tokenizer configuration from fine-tuning
    print(f"Loading tokenizer from: {LORA_ADAPTER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})")


    # 2. Load the base model
    print(f"Loading base model: {BASE_MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32 # bfloat16 for GPU, float32 for CPU
    )
    base_model.to(device)
    print("Base model loaded.")

    # 3. Load and apply the LoRA adapter
    print(f"Loading LoRA adapter from: {LORA_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    model = model.to(device)
    model.eval() # Set the model to evaluation mode
    print("LoRA adapter loaded and applied.")

    # 4. Prepare a prompt using the command-line argument
    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT_CONTENT},
        {"role": "user", "content": user_query}
    ]

    # --- First model call ---
    prompt_for_model = ""
    for turn in conversation_history:
        prompt_for_model += f"{IM_START}{turn['role']}\n{turn['content']}{IM_END}\n"
    prompt_for_model += f"{IM_START}assistant\n" # Model should continue from here
    
    # print(f"\n--- Prompt for 1st model call ---\n{prompt_for_model}")

    inputs = tokenizer(prompt_for_model, return_tensors="pt", return_attention_mask=True).to(device)

    print("\nGenerating initial response (potential tool call)...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            temperature=0.1, # Lower temperature for more deterministic tool calls
            top_p=0.9,
            do_sample=True, # Still sample, but low temp makes it more focused
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(IM_END)],
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
    
    assistant_response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    # Remove any leading/trailing whitespace or newlines that are not part of the structured response
    assistant_response_text = assistant_response_text.strip()

    print(f"\nAssistant's raw output (1st call):\n{assistant_response_text}")

    # Add assistant's response to conversation history (even if it's a tool call)
    # We need to be careful not to add the <|im_end|> if it's already there from generation
    # For now, let's assume the model generates the content and then we add the structure
    
    # Attempt to parse for function call
    tool_call_data = extract_json_from_response(assistant_response_text)

    if tool_call_data and "function_call" in tool_call_data:
        function_call_details = tool_call_data["function_call"]
        tool_name = function_call_details.get("name")
        tool_args_str = function_call_details.get("arguments") # This should be a JSON string

        if not isinstance(tool_args_str, str):
            print(f"Warning: Tool arguments are not a string: {tool_args_str}")
            tool_args_str = json.dumps(tool_args_str) # Attempt to stringify if it's a dict

        print(f"\n--- Detected Tool Call ---")
        print(f"Tool Name: {tool_name}")
        print(f"Tool Arguments (raw string): {tool_args_str}")

        # Call the tool executor script
        try:
            process = subprocess.run(
                [sys.executable, TOOL_EXECUTOR_SCRIPT, tool_name, tool_args_str],
                capture_output=True,
                text=True,
                check=True
            )
            tool_result_json_str = process.stdout.strip()
            print(f"Tool Execution Result (JSON string): {tool_result_json_str}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing tool: {e}")
            print(f"Stderr: {e.stderr}")
            tool_result_json_str = json.dumps({"status": "error", "message": f"Failed to execute tool: {e.stderr or e}"})
        except FileNotFoundError:
            print(f"Error: Tool executor script '{TOOL_EXECUTOR_SCRIPT}' not found.")
            tool_result_json_str = json.dumps({"status": "error", "message": f"Tool executor script not found."})
        
        # --- Second model call with tool result ---
        # Add the assistant's first response (the tool call itself) to history
        # The model's output might already include <|im_end|>, strip it if so before adding our own.
        clean_assistant_response = assistant_response_text
        if clean_assistant_response.endswith(IM_END):
            clean_assistant_response = clean_assistant_response[:-len(IM_END)].strip()
        
        conversation_history.append({"role": "assistant", "content": clean_assistant_response})
        conversation_history.append({"role": "function", "name": tool_name, "content": tool_result_json_str}) # Content is the JSON string result

        prompt_for_2nd_call = ""
        for turn in conversation_history:
            if turn["role"] == "function":
                 prompt_for_2nd_call += f"{IM_START}function\n{json.dumps({'name': turn['name'], 'content': turn['content']})}{IM_END}\n"
            else:
                prompt_for_2nd_call += f"{IM_START}{turn['role']}\n{turn['content']}{IM_END}\n"
        prompt_for_2nd_call += f"{IM_START}assistant\n"

        # print(f"\n--- Prompt for 2nd model call ---\n{prompt_for_2nd_call}")
        
        inputs_2nd_call = tokenizer(prompt_for_2nd_call, return_tensors="pt", return_attention_mask=True).to(device)
        print("\nGenerating final response after tool execution...")
        with torch.no_grad():
            outputs_2nd_call = model.generate(
                input_ids=inputs_2nd_call["input_ids"],
                attention_mask=inputs_2nd_call["attention_mask"],
                max_new_tokens=100, # Usually shorter for summarizing tool output
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
        final_response_text = tokenizer.decode(outputs_2nd_call[0][inputs_2nd_call["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\nFinal Natural Language Response:\n{final_response_text.strip()}")

    else:
        print("\nNo tool call detected in model's initial response.")
        # Just print the assistant's response as is (after stripping potential <|im_end|>)
        final_response_text = assistant_response_text.replace(IM_END, "").strip()
        print(f"Model's direct response:\n{final_response_text}")


if __name__ == "__main__":
    main()