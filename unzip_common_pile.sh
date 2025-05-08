#!/bin/bash

source_dir="/p/vast1/pretrain/datasets/common_pile/common-pile-chunked"
output_dir="/p/vast1/pretrain/datasets/common_pile/common-pile-chunked-unzipped"
num_threads="32"

# Function to process each .gz file
process_gzip() {
  local gz_file="$1"
  local relative_path="${gz_file#"$source_dir/"}"
  local output_file="$output_dir/${relative_path%.gz}"
  local output_dir_path="$(dirname "$output_file")"

  # Create the output directory if it doesn't exist
  mkdir -p "$output_dir_path"

  # Decompress the file using pigz
  pigz -dcv "$gz_file" > "$output_file"
  echo "Unzipped: $gz_file -> $output_file"
}

export -f process_gzip

# Find all .gz files under the source directory and process them in parallel
find "$source_dir" -type f -name "*.gz" -print0 | \
  xargs -0 -n 1 -P "$num_threads" bash -c 'process_gzip "$0"'


# find common-pile-chunked -name "*.gz" -print0 | while IFS= read -r -d $'\0' file; do
#   dest_dir=$(dirname "${file#source_dir}")
#   mkdir -p "/p/vast1/pretrain/datasets/common_pile/common-pile-chunked-unzipped/${dest_dir}"
#   gunzip -cv "$file" > "/p/vast1/pretrain/datasets/common_pile/common-pile-chunked-unzipped/${dest_dir}/$(basename "${file%.gz}")"
# done
