#!/bin/bash
# Shell script: run_intruder_detection_batch.sh

# Path to the file containing the file paths
file_list="file_paths.txt"

# Path to the Python script
python_script_path="./main.py"

# Read each line from the file list
while IFS= read -r line; do

    # Skip blank lines
    if [ -z "$line" ]; then
        continue
    fi

    # Split the line into an array of file paths
    file_paths=($line)
    
    # Extract a common prefix from the directory path of the first file path
    common_prefix=$(echo "$line" | cut -d'/' -f1-4 | tr '/' '_')
    common_prefix="cloud_script_results/${common_prefix}"

    # Call the Python script with the input files
    python3 "$python_script_path" "${file_paths[@]}"

    # Check if the results.txt file was created and rename it to include the common prefix
    if [ -f results.txt ]; then
        output_dir="cloud_script_results/"
        mkdir -p "$output_dir"
        mv results.txt "${common_prefix}_results.txt"
        echo "Results saved to ${common_prefix}_results.txt"
    else
        echo "Error: The results file was not created for set: $line"
    fi
done < "$file_list"
