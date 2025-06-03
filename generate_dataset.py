import json
import random
import os

# Base system prompt
SYSTEM_PROMPT = "You are a helpful AI assistant. You are capable of file system operations (navigate, list, move, copy, delete) and running applications by calling tools. When you decide to use a tool, first think about the steps and then call the tool using the specified JSON format."

# Helper to create the "text" field content
def create_chat_entry(user_query, assistant_think_content, assistant_tool_call_json, function_name, function_response_content, assistant_final_response):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>{assistant_think_content}</think>{json.dumps(assistant_tool_call_json)}<|im_end|>\n"
        f"<|im_start|>function\n{json.dumps({'name': function_name, 'content': json.dumps(function_response_content)})}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_final_response}<|im_end|>"
    )

# --- Generation functions for each tool ---

def generate_list_files_examples(num_examples=50):
    examples = []
    common_paths = ["documents", "downloads", "projects/alpha", "src/utils", "temp/logs", ".", "C:/Users/Default/Desktop", "my work/important files"]
    common_files_dirs = [
        ["report.docx", "image.png", "data_folder"],
        ["script.py", "notes.txt"],
        [], # Empty directory
        ["archive.zip", "backup.tar.gz", "old_docs"],
        ["main.c", "Makefile", "include_dir", "README.md"]
    ]

    for i in range(num_examples):
        path_to_list = random.choice(common_paths)
        if path_to_list == ".": # Implies current directory
            user_query_phrases = [
                f"What files are in my current location?",
                f"Show me the contents here.",
                f"List everything in the current directory.",
            ]
            think_content = "The user wants to list files in the current directory. I should use the 'list_files' tool. No path argument is needed for the current directory."
            tool_args = {}
        else:
            user_query_phrases = [
                f"Show me what files are in the '{path_to_list}' folder?",
                f"Can you list the contents of '{path_to_list}'?",
                f"What's inside '{path_to_list}'?",
                f"List '{path_to_list}'.",
            ]
            think_content = f"The user wants to list files in '{path_to_list}'. I should use the 'list_files' tool with the path '{path_to_list}'."
            tool_args = {"path": path_to_list}

        user_query = random.choice(user_query_phrases)
        assistant_tool_call = {"function_call": {"name": "list_files", "arguments": json.dumps(tool_args)}}

        # Simulate success or error
        if random.random() < 0.1 and path_to_list != ".": # 10% chance of error for non-current paths
            error_type = random.choice(["not_found", "permission_denied"])
            if error_type == "not_found":
                function_response = {"status": "error", "message": f"Error: Path {path_to_list} not found."}
                assistant_response = f"I couldn't find the folder '{path_to_list}'. Please check if the path is correct."
            else: # permission_denied
                function_response = {"status": "error", "message": f"Error: Permission denied for {path_to_list}."}
                assistant_response = f"I'm sorry, I don't have permission to access the folder '{path_to_list}'."
        else:
            listed_items = random.choice(common_files_dirs)
            function_response = {"status": "success", "files": listed_items}
            if not listed_items:
                assistant_response = f"The folder '{path_to_list if path_to_list != '.' else 'current directory'}' appears to be empty."
            else:
                assistant_response = f"In '{path_to_list if path_to_list != '.' else 'the current directory'}', I found: {', '.join(listed_items)}."

        examples.append({"text": create_chat_entry(user_query, think_content, assistant_tool_call, "list_files", function_response, assistant_response)})
    return examples

def generate_navigate_directory_examples(num_examples=50):
    examples = []
    # TODO: Implement diverse example generation
    # For now, a placeholder:
    for _ in range(num_examples):
        path = f"projects/project_{random.randint(1,100)}"
        user_query = f"Navigate to {path}"
        think = f"User wants to navigate to {path}. Use 'navigate_directory'."
        tool_call = {"function_call": {"name": "navigate_directory", "arguments": json.dumps({"path": path})}}
        func_resp = {"status": "success", "message": f"Current directory changed to {path}"}
        assistant_resp = f"Okay, I've navigated to '{path}'."
        examples.append({"text": create_chat_entry(user_query, think, tool_call, "navigate_directory", func_resp, assistant_resp)})
    return examples

def generate_copy_file_examples(num_examples=50):
    examples = []
    # TODO: Implement diverse example generation
    for _ in range(num_examples):
        src = f"source_folder/file_{random.randint(1,100)}.txt"
        dest = f"destination_folder/file_copy_{random.randint(1,100)}.txt"
        user_query = f"Copy {src} to {dest}"
        think = f"User wants to copy. Source: {src}, Dest: {dest}. Use 'copy_file'."
        tool_call = {"function_call": {"name": "copy_file", "arguments": json.dumps({"source_path": src, "destination_path": dest})}}
        func_resp = {"status": "success", "message": f"File {src} copied to {dest}"}
        assistant_resp = f"I've copied '{src}' to '{dest}'."
        examples.append({"text": create_chat_entry(user_query, think, tool_call, "copy_file", func_resp, assistant_resp)})
    return examples

def generate_move_file_examples(num_examples=50):
    examples = []
    # TODO: Implement diverse example generation
    for _ in range(num_examples):
        src = f"old_location/item_{random.randint(1,100)}.dat"
        dest = f"new_archive/item_moved_{random.randint(1,100)}.dat"
        user_query = f"Move {src} to {dest}"
        think = f"User wants to move. Source: {src}, Dest: {dest}. Use 'move_file'."
        tool_call = {"function_call": {"name": "move_file", "arguments": json.dumps({"source_path": src, "destination_path": dest})}}
        func_resp = {"status": "success", "message": f"File {src} moved to {dest}"}
        assistant_resp = f"Done. '{src}' has been moved to '{dest}'."
        examples.append({"text": create_chat_entry(user_query, think, tool_call, "move_file", func_resp, assistant_resp)})
    return examples

def generate_delete_file_examples(num_examples=50):
    examples = []
    # TODO: Implement diverse example generation
    for _ in range(num_examples):
        file_to_delete = f"temp_files/obsolete_{random.randint(1,100)}.tmp"
        user_query = f"Delete {file_to_delete}"
        think = f"User wants to delete {file_to_delete}. Use 'delete_file'."
        tool_call = {"function_call": {"name": "delete_file", "arguments": json.dumps({"path": file_to_delete})}}
        func_resp = {"status": "success", "message": f"File {file_to_delete} deleted."}
        assistant_resp = f"I have deleted '{file_to_delete}'."
        examples.append({"text": create_chat_entry(user_query, think, tool_call, "delete_file", func_resp, assistant_resp)})
    return examples

def generate_run_application_examples(num_examples=50):
    examples = []
    apps = ["calculator", "notepad", "cmd", "python_script.py --version"]
    # TODO: Implement diverse example generation, including arguments
    for _ in range(num_examples):
        app_name = random.choice(apps)
        args_json = "{}" # Default for no args
        think_content = f"User wants to run '{app_name}'. Use 'run_application'." # Default think
        
        app_name_actual = app_name # Actual app name for tool call
        tool_call_args_dict = {"application_name": app_name_actual}

        if " " in app_name: # simple argument parsing for this example
            parts = app_name.split(" ", 1)
            app_name_actual = parts[0]
            args_list = parts[1].split()
            tool_call_args_dict = {"application_name": app_name_actual, "arguments": args_list}
            think_content = f"User wants to run '{app_name_actual}' with arguments '{' '.join(args_list)}'. Use 'run_application'."
        
        user_query = f"Run {app_name}" # User query uses the full app string if it has args

        # assistant_tool_call_json expects the full function_call structure
        assistant_tool_call = {"function_call": {"name": "run_application", "arguments": json.dumps(tool_call_args_dict)}}
        
        func_resp = {"status": "success", "message": f"Application {app_name_actual} started."}
        assistant_resp = f"I've started the '{app_name_actual}' application for you."
        examples.append({"text": create_chat_entry(user_query, think_content, assistant_tool_call, "run_application", func_resp, assistant_resp)})
    return examples

def main():
    output_file = "tool_training_dataset.jsonl"
    all_examples = []

    print("Generating 'list_files' examples...")
    all_examples.extend(generate_list_files_examples(50))
    print("Generating 'navigate_directory' examples...")
    all_examples.extend(generate_navigate_directory_examples(50))
    print("Generating 'copy_file' examples...")
    all_examples.extend(generate_copy_file_examples(50))
    print("Generating 'move_file' examples...")
    all_examples.extend(generate_move_file_examples(50))
    print("Generating 'delete_file' examples...")
    all_examples.extend(generate_delete_file_examples(50))
    print("Generating 'run_application' examples...")
    all_examples.extend(generate_run_application_examples(50))

    # Ensure the directory for the output file exists
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    with open(output_file, "w") as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")

    print(f"Generated {len(all_examples)} examples in {output_file}")

if __name__ == "__main__":
    main()