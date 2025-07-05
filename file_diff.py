#!/usr/bin/env python3
"""
Simple file difference finder for .txt files
Usage: python file_diff.py file1.txt file2.txt
"""

import sys
import difflib
import os

def compare_files(file1_path, file2_path):
    """Compare two text files and display their differences."""
    
    # Check if both files exist
    if not os.path.exists(file1_path):
        print(f"Error: File '{file1_path}' not found.")
        return False
    
    if not os.path.exists(file2_path):
        print(f"Error: File '{file2_path}' not found.")
        return False
    
    try:
        # Read the files
        with open(file1_path, 'r', encoding='utf-8') as f1:
            file1_lines = f1.readlines()
        
        with open(file2_path, 'r', encoding='utf-8') as f2:
            file2_lines = f2.readlines()
        
        # Generate the diff
        diff = difflib.unified_diff(
            file1_lines,
            file2_lines,
            fromfile=file1_path,
            tofile=file2_path,
            lineterm=''
        )
        
        # Check if there are any differences
        diff_lines = list(diff)
        
        if not diff_lines:
            print(f"No differences found between '{file1_path}' and '{file2_path}'")
            return True
        
        # Print the differences
        print(f"Differences between '{file1_path}' and '{file2_path}':")
        print("=" * 60)
        
        for line in diff_lines:
            print(line)
        
        return True
        
    except Exception as e:
        print(f"Error reading files: {e}")
        return False

def main():
    """Main function to handle command line arguments."""
    
    if len(sys.argv) != 3:
        print("Usage: python file_diff.py file1.txt file2.txt")
        print("Example: python file_diff.py document1.txt document2.txt")
        sys.exit(1)
    
    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    
    print(f"Comparing files: '{file1_path}' and '{file2_path}'")
    print()
    
    success = compare_files(file1_path, file2_path)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 