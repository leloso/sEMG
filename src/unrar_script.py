import os
import sys
import subprocess
import shutil

def unrar_file(rar_filepath, output_dir, delete_after_unrar=False):
    """
    Extracts a single .rar file to the specified output directory.
    If delete_after_unrar is True, the .rar file will be deleted upon successful extraction.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        # Check if unrar command is available
        subprocess.run(['unrar', '-v'], check=True, capture_output=True, text=True)
    except FileNotFoundError:
        print("Error: 'unrar' command not found.")
        print("Please install 'unrar' on your system. On Debian/Ubuntu: sudo apt-get install unrar")
        print("On macOS (with Homebrew): brew install unrar")
        print("On Windows, download from https://www.rarlab.com/rar_add.htm and add to PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error: 'unrar' command found, but returned an error on version check (Code: {e.returncode}). Stderr: {e.stderr.strip()}")
        return False

    print(f"Attempting to unrar: {rar_filepath} to {output_dir}")
    try:
        result = subprocess.run(
            ['unrar', 'x', '-o-', rar_filepath, output_dir],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Successfully unrared: {rar_filepath}")
        
        if delete_after_unrar:
            try:
                os.remove(rar_filepath)
                print(f"Deleted original RAR file: {rar_filepath}")
                # If it's a multi-part RAR, you might want to delete all parts.
                # This requires more complex logic to identify all parts (e.g., .r00, .r01, .part1.rar, etc.)
                # For simplicity, this script only deletes the main .rar file provided.
            except OSError as e:
                print(f"Error deleting {rar_filepath}: {e}")
                # Extraction was successful, but deletion failed, so still return True for extraction success
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error unrarring {rar_filepath}:")
        print(f"  Return Code: {e.returncode}")
        print(f"  Stdout: {e.stdout.strip()}")
        print(f"  Stderr: {e.stderr.strip()}")
        if "No files to extract" in e.stderr:
            print("  (This might mean the RAR is empty or corrupted, or all files already exist and were skipped.)")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while processing {rar_filepath}: {e}")
        return False

def recursively_unrar_directory(base_directory, delete_after_unrar_flag=False):
    """
    Recursively finds and unrars .rar files within a given directory.
    If delete_after_unrar_flag is True, .rar files will be deleted after successful unrar.
    """
    if not os.path.isdir(base_directory):
        print(f"Error: Directory not found: {base_directory}")
        return

    print(f"Starting recursive unrar in: {base_directory}")
    rar_found_count = 0
    rar_extracted_count = 0
    rar_deleted_count = 0

    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.lower().endswith('.rar'):
                rar_found_count += 1
                rar_filepath = os.path.join(root, file)
                output_dir = root # Extract to the same directory as the .rar file
                
                print(f"\nFound RAR file: {rar_filepath}")
                if unrar_file(rar_filepath, output_dir, delete_after_unrar=delete_after_unrar_flag):
                    rar_extracted_count += 1
                    # If unrar_file successfully deleted it, it prints a message.
                    # We just need to increment our script's overall deleted count if deletion was attempted.
                    if delete_after_unrar_flag and not os.path.exists(rar_filepath):
                        rar_deleted_count += 1
                else:
                    print(f"Failed to unrar: {rar_filepath}")

    print(f"\n--- Unrar Summary ---")
    print(f"Total .rar files found: {rar_found_count}")
    print(f"Total .rar files successfully extracted: {rar_extracted_count}")
    if delete_after_unrar_flag:
        print(f"Total .rar files successfully deleted: {rar_deleted_count}")
    print(f"Total .rar files failed to extract: {rar_found_count - rar_extracted_count}")

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python unrar_script.py <directory_path> [--delete]")
        print("  <directory_path>: The root directory to start unraring from.")
        print("  --delete: Optional flag. If present, .rar files will be deleted after successful extraction.")
        sys.exit(1)

    target_directory = sys.argv[1]
    delete_flag = '--delete' in sys.argv[1:] # Check if '--delete' is anywhere in arguments

    recursively_unrar_directory(target_directory, delete_after_unrar_flag=delete_flag)