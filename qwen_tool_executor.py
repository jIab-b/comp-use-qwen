import json
import os
import shutil
import argparse

def list_files(path="."):
    """
    Lists files and directories at a given path.
    If path is empty or '.', lists contents of the current working directory.
    """
    try:
        if not path: # Handle empty path string as current directory
            path = "."
        
        if not os.path.exists(path):
            return json.dumps({"status": "error", "message": f"Error: Path '{path}' not found."})
        if not os.path.isdir(path):
            return json.dumps({"status": "error", "message": f"Error: Path '{path}' is not a directory."})
        
        items = os.listdir(path)
        return json.dumps({"status": "success", "files": items})
    except PermissionError:
        return json.dumps({"status": "error", "message": f"Error: Permission denied for path '{path}'."})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

def move_file_system(source_path, destination_path):
    """
    Moves a file or directory from source_path to destination_path.
    """
    try:
        if not source_path or not destination_path:
            return json.dumps({"status": "error", "message": "Error: Source and destination paths must be provided."})
        
        if not os.path.exists(source_path):
            return json.dumps({"status": "error", "message": f"Error: Source path '{source_path}' not found."})
        
        # Ensure destination directory exists if destination_path is a directory
        dest_dir = os.path.dirname(destination_path)
        if dest_dir and not os.path.exists(dest_dir):
            # If destination_path looks like a directory (e.g., ends with / or is an existing dir)
            # and its parent doesn't exist, this is an issue.
            # However, shutil.move can create the final directory if it's part of the dest name.
            # For simplicity, we'll let shutil.move handle it, but more robust checks could be added.
            pass

        shutil.move(source_path, destination_path)
        return json.dumps({"status": "success", "message": f"Successfully moved '{source_path}' to '{destination_path}'."})
    except FileNotFoundError: # Should be caught by os.path.exists, but as a fallback
        return json.dumps({"status": "error", "message": f"Error: Source path '{source_path}' not found."})
    except shutil.Error as e: # Catches errors like destination already exists, permission issues etc.
         return json.dumps({"status": "error", "message": f"shutil.Error: {str(e)}"})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

# Add other tool functions here as needed (e.g., copy_file, delete_file, run_application)

AVAILABLE_TOOLS = {
    "list_files": list_files,
    "move_file_system": move_file_system,
    # "copy_file": copy_file, # Example for future
    # "delete_file": delete_file, # Example for future
}

def main():
    parser = argparse.ArgumentParser(description="Qwen Tool Executor: Executes a specified tool with given arguments.")
    parser.add_argument("tool_name", help="The name of the tool to execute.", choices=AVAILABLE_TOOLS.keys())
    parser.add_argument("tool_args_json", help="A JSON string containing the arguments for the tool.")

    args = parser.parse_args()

    tool_function = AVAILABLE_TOOLS.get(args.tool_name)
    
    if not tool_function:
        # This case should ideally not be reached if choices are enforced by argparse
        print(json.dumps({"status": "error", "message": f"Error: Tool '{args.tool_name}' not found."}))
        return

    try:
        tool_arguments = json.loads(args.tool_args_json)
        # Ensure path arguments are treated as relative to the current working directory
        # if they are not absolute paths. This is important for consistency.
        if args.tool_name == "list_files" and "path" in tool_arguments:
            tool_arguments["path"] = os.path.abspath(tool_arguments.get("path", "."))
        elif args.tool_name == "move_file_system":
            if "source_path" in tool_arguments:
                 tool_arguments["source_path"] = os.path.abspath(tool_arguments["source_path"])
            if "destination_path" in tool_arguments:
                 tool_arguments["destination_path"] = os.path.abspath(tool_arguments["destination_path"])
        
        result = tool_function(**tool_arguments)
    except json.JSONDecodeError:
        result = json.dumps({"status": "error", "message": "Error: Invalid JSON provided for tool arguments."})
    except TypeError as e: # Handles issues like missing arguments for the tool function
        result = json.dumps({"status": "error", "message": f"Error calling tool '{args.tool_name}': {str(e)}"})
    except Exception as e: # Catch-all for other unexpected errors during tool execution
        result = json.dumps({"status": "error", "message": f"Unexpected error executing tool '{args.tool_name}': {str(e)}"})
        
    print(result)

if __name__ == "__main__":
    main()