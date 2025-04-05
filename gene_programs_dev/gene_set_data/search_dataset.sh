#!/bin/bash

# SCRIPT TO SEARCH DIRECTORIES THAT START WITH A SPECIFIED KEYWORD

# Base directory to search
BASE_DIR="/home/ddz5/Desktop/c2s-RL/gene_programs_dev/gene_set_data"

# Check if a keyword was provided
if [ $# -eq 0 ]; then
    echo "Please provide a search keyword as an argument"
    echo "Usage: $0 <search_keyword>"
    echo "Example: $0 'local(258)'"
    exit 1
fi

# Get the search keyword from command line argument
SEARCH_KEYWORD="$1"
# Escape special characters in the search keyword for regex
SEARCH_PATTERN=$(echo "$SEARCH_KEYWORD" | sed 's/[][().*+?^$\\\/]/\\&/g')

# Output file for results
OUTPUT_FILE="found_dirs.txt"
> "$OUTPUT_FILE"  # Initialize/clear the output file

# Counter for found directories
found_count=0

# Function to search for directories recursively
find_local_dirs() {
    local current_dir="$1"
    local indent="$2"
    
    # Process items in current directory
    for item in "$current_dir"/*; do
        # Skip if item doesn't exist
        [ ! -e "$item" ] && continue
        
        # Get the basename of the item
        base_name=$(basename "$item")
        
        if [ -d "$item" ]; then
            # Check if directory name matches the pattern
            if [[ "$base_name" == ${SEARCH_PATTERN}* ]]; then
                echo "${indent}FOUND: $item"
                echo "$item" >> "$OUTPUT_FILE"
                ((found_count++))
            fi
            
            # Recursively search subdirectories
            find_local_dirs "$item" "  $indent"
        fi
    done
}

echo "Starting search for directories starting with '$SEARCH_KEYWORD' in $BASE_DIR..."
find_local_dirs "$BASE_DIR" ""

# Print summary
echo "Search complete!"
if [ $found_count -eq 0 ]; then
    echo "No directories matching '$SEARCH_KEYWORD*' were found."
else
    echo "Found $found_count matching directories."
    echo "Results have been saved to $OUTPUT_FILE"
fi
