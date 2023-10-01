import json
import os
import sys

import pdal


def file_format(file_path):
    with open('ply_binary.json', 'r') as file:
        json_data = json.load(file)

    # Input file path, readers.las -> filename
    json_data[0]["filename"] = file_path

    # Output file path, writers.ply -> filename
    json_data[1]["filename"] = os.path.join(input_path, base_name + ".ply")

    # Convert the JSON object to a string
    json_string = json.dumps(json_data)

    # Execute PDAL pipeline
    pipeline = pdal.Pipeline(json_string)
    pipeline.execute()


if __name__ == "__main__":
    # File directory
    input_path = "../pipeline/data/input"
    os.makedirs(input_path, exist_ok=True)

    # List of file paths to process
    if len(sys.argv) > 1:
        file_list = sys.argv[1:]
    else:
        print("Warning: No input filenames passed.")
        exit(-1)

    for index, input_file_path in enumerate(file_list):
        # Prompt current file path
        base_name, extension = os.path.splitext(os.path.basename(input_file_path))
        print(f"File format conversion [{index + 1}]: {input_file_path}")
        file_format(input_file_path)
