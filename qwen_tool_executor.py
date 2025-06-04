import json
import os
import shutil
import argparse
import subprocess # Added for run_application

def list_files(path="."):
    """
    Lists files and directories at a given path.
    If path is empty or '.', lists contents of the current working directory.
    """
    try:
        # Path normalization will be handled in main for consistency
        # if not path: path = "."
        
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

def move_file(source_path, destination_path): # Renamed from move_file_system
    """
    Moves a file or directory from source_path to destination_path.
    """
    try:
        if not source_path or not destination_path:
            return json.dumps({"status": "error", "message": "Error: Source and destination paths must be provided."})
        
        if not os.path.exists(source_path):
            return json.dumps({"status": "error", "message": f"Error: Source path '{source_path}' not found."})
        
        dest_dir = os.path.dirname(destination_path)
        if dest_dir and not os.path.exists(dest_dir) and not os.path.isfile(destination_path): # Check if dest is not a file path
             os.makedirs(dest_dir, exist_ok=True)


        shutil.move(source_path, destination_path)
        return json.dumps({"status": "success", "message": f"Successfully moved '{source_path}' to '{destination_path}'."})
    except FileNotFoundError:
        return json.dumps({"status": "error", "message": f"Error: Source path '{source_path}' not found."})
    except shutil.Error as e:
         return json.dumps({"status": "error", "message": f"shutil.Error: {str(e)}"})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

def navigate_directory(path):
    """
    Navigates to the specified directory path.
    """
    try:
        if not path:
            return json.dumps({"status": "error", "message": "Error: Path must be provided for navigation."})
        
        if not os.path.exists(path):
            return json.dumps({"status": "error", "message": f"Error: Path '{path}' not found."})
        if not os.path.isdir(path):
            return json.dumps({"status": "error", "message": f"Error: Path '{path}' is not a directory."})
        
        os.chdir(path)
        return json.dumps({"status": "success", "message": f"Current directory changed to '{os.getcwd()}'."})
    except FileNotFoundError:
        return json.dumps({"status": "error", "message": f"Error: Path '{path}' not found."})
    except NotADirectoryError:
         return json.dumps({"status": "error", "message": f"Error: Path '{path}' is not a directory."})
    except PermissionError:
        return json.dumps({"status": "error", "message": f"Error: Permission denied for path '{path}'."})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

def copy_file(source_path, destination_path):
    """
    Copies a file from source_path to destination_path.
    """
    try:
        if not source_path or not destination_path:
            return json.dumps({"status": "error", "message": "Error: Source and destination paths must be provided."})
        
        if not os.path.exists(source_path):
            return json.dumps({"status": "error", "message": f"Error: Source path '{source_path}' not found."})
        if not os.path.isfile(source_path):
            return json.dumps({"status": "error", "message": f"Error: Source path '{source_path}' is not a file."})

        dest_dir = os.path.dirname(destination_path)
        if dest_dir and not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)

        shutil.copy2(source_path, destination_path)
        return json.dumps({"status": "success", "message": f"Successfully copied '{source_path}' to '{destination_path}'."})
    except FileNotFoundError:
        return json.dumps({"status": "error", "message": f"Error: Source path '{source_path}' not found."})
    except shutil.SameFileError:
        return json.dumps({"status": "error", "message": f"Error: Source and destination are the same file: '{source_path}'."})
    except PermissionError:
        return json.dumps({"status": "error", "message": f"Error: Permission denied during copy operation."})
    except IsADirectoryError:
         return json.dumps({"status": "error", "message": f"Error: Source is a directory or destination is invalid."})
    except shutil.Error as e:
        return json.dumps({"status": "error", "message": f"shutil.Error: {str(e)}"})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

def delete_file(path):
    """
    Deletes a file at the specified path.
    """
    try:
        if not path:
            return json.dumps({"status": "error", "message": "Error: Path must be provided for deletion."})

        if not os.path.exists(path):
            return json.dumps({"status": "error", "message": f"Error: File '{path}' not found."})
        if not os.path.isfile(path):
             return json.dumps({"status": "error", "message": f"Error: Path '{path}' is not a file. This function only deletes files."})

        os.remove(path)
        return json.dumps({"status": "success", "message": f"Successfully deleted file '{path}'."})
    except FileNotFoundError:
        return json.dumps({"status": "error", "message": f"Error: File '{path}' not found."})
    except PermissionError:
        return json.dumps({"status": "error", "message": f"Error: Permission denied for file '{path}'."})
    except IsADirectoryError:
        return json.dumps({"status": "error", "message": f"Error: Path '{path}' is a directory, not a file."})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

def run_application(application_name, arguments=None):
    """
    Runs an application with optional arguments.
    'arguments' should be a list of strings.
    """
    try:
        if not application_name:
            return json.dumps({"status": "error", "message": "Error: Application name must be provided."})

        command = [application_name]
        if arguments:
            if not isinstance(arguments, list):
                # Attempt to parse if it's a stringified list, though dataset should provide a list
                if isinstance(arguments, str):
                    try:
                        parsed_args = json.loads(arguments)
                        if isinstance(parsed_args, list):
                            arguments = parsed_args
                        else:
                            return json.dumps({"status": "error", "message": "Error: Arguments, if a string, must be a JSON list of strings."})
                    except json.JSONDecodeError:
                         return json.dumps({"status": "error", "message": "Error: Arguments string is not valid JSON."})
                else:
                    return json.dumps({"status": "error", "message": "Error: Arguments must be a list of strings or a JSON string list."})
            command.extend(arguments)
        
        process = subprocess.run(command, capture_output=True, text=True, check=False)

        if process.returncode == 0:
            return json.dumps({
                "status": "success",
                "message": f"Application '{application_name}' executed successfully.",
                "stdout": process.stdout.strip(),
                "stderr": process.stderr.strip()
            })
        else:
            return json.dumps({
                "status": "error",
                "message": f"Application '{application_name}' exited with error code {process.returncode}.",
                "stdout": process.stdout.strip(),
                "stderr": process.stderr.strip()
            })
    except FileNotFoundError:
        return json.dumps({"status": "error", "message": f"Error: Application '{application_name}' not found. Make sure it's in PATH or provide full path."})
    except PermissionError:
        return json.dumps({"status": "error", "message": f"Error: Permission denied when trying to run '{application_name}'."})
    except subprocess.TimeoutExpired:
         return json.dumps({"status": "error", "message": f"Error: Application '{application_name}' timed out."})
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Error running application '{application_name}': {str(e)}"})

AVAILABLE_TOOLS = {
    "list_files": list_files,
    "move_file": move_file, # Renamed
    "navigate_directory": navigate_directory,
    "copy_file": copy_file,
    "delete_file": delete_file,
    "run_application": run_application,
}

def main():
    parser = argparse.ArgumentParser(description="Qwen Tool Executor: Executes a specified tool with given arguments.")
    parser.add_argument("tool_name", help="The name of the tool to execute.", choices=AVAILABLE_TOOLS.keys())
    parser.add_argument("tool_args_json", help="A JSON string containing the arguments for the tool.")

    args = parser.parse_args()

    tool_function = AVAILABLE_TOOLS.get(args.tool_name)
    
    if not tool_function:
        print(json.dumps({"status": "error", "message": f"Error: Tool '{args.tool_name}' not found."}))
        return

    try:
        tool_arguments = json.loads(args.tool_args_json)
        
        # Path argument processing
        if args.tool_name == "list_files":
            path_arg = tool_arguments.get("path", ".") # Default to "." if not provided or empty
            if not path_arg: path_arg = "." # Ensure empty string becomes "."
            tool_arguments["path"] = os.path.abspath(path_arg)
        elif args.tool_name == "navigate_directory":
            if "path" in tool_arguments and tool_arguments["path"]:
                tool_arguments["path"] = os.path.abspath(tool_arguments["path"])
            else: # Path is mandatory for navigate, function will handle error
                pass
        elif args.tool_name == "delete_file":
            if "path" in tool_arguments and tool_arguments["path"]:
                tool_arguments["path"] = os.path.abspath(tool_arguments["path"])
            else: # Path is mandatory
                pass
        elif args.tool_name == "move_file" or args.tool_name == "copy_file":
            if "source_path" in tool_arguments and tool_arguments["source_path"]:
                 tool_arguments["source_path"] = os.path.abspath(tool_arguments["source_path"])
            if "destination_path" in tool_arguments and tool_arguments["destination_path"]:
                 tool_arguments["destination_path"] = os.path.abspath(tool_arguments["destination_path"])
        # No abspath for run_application args or app_name itself.
        
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