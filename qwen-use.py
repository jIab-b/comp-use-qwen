from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run Qwen model with a given prompt.")
    parser.add_argument("prompt", type=str, help="The prompt to send to the Qwen model.")
    
    args = parser.parse_args()

    model_name = "Qwen/Qwen3-0.6B"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    inputs = tokenizer(args.prompt, return_tensors="pt")
    
    try:
        outputs = model.generate(**inputs, max_new_tokens=300)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    except Exception as e:
        print(f"Error during model generation: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
