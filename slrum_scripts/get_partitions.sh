#!/bin/bash

# Check if a directory is provided
if [ -z "$1" ]; then
  echo "Please provide the directory path as an argument."
  exit 1
fi

# Directory path
dir_path="$1"

# List the files in the specified directory and extract numbers using awk and sed
numbers=$(ls -1 "$dir_path" | grep 'corpus_strict_v3.0_en_compiled_passages_lang_' | awk -F'lang_' '{print $2}' | sed 's/\.parquet//')
#numbers=$(ls -1 "$dir_path" | grep 'corpus_strict_v2.0_es_compiled_passages_lang_' | awk -F'lang_' '{print $2}' | sed 's/\.parquet//')


# Convert the numbers to an array and sort them
numbers_array=($(echo "$numbers" | sort -n))

# Get the first and last elements of the sorted array
first_number=${numbers_array[0]}
last_number=${numbers_array[-1]}

# Print the first and last numbers
echo "First number: $first_number"
echo "Last number: $last_number"

# sh get_partitions.sh /fs/nexus-scratch/lcalvo/rosie/data/en