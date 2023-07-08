import json
import os
import sys

import pdal


def file_format(filename):
    with open('ply_binary.json', 'r') as file:
        json_data = json.load(file)

    # Input filename, readers.las -> filename
    json_data[0]["filename"] = filename

    output_filepath = "../data/input"
    os.makedirs(output_filepath, exist_ok=True)

    base_name, _ = os.path.splitext(os.path.basename(filename))
    output_filename = base_name + ".ply"

    # Output filename, writers.ply -> filename
    json_data[1]["filename"] = os.path.join(output_filepath, output_filename)

    # Convert the JSON object to a string
    json_string = json.dumps(json_data)

    pipeline = pdal.Pipeline(json_string)
    pipeline.execute()


if __name__ == "__main__":
    filenames = []

    if len(sys.argv) > 1:
        filenames = sys.argv[1:]
    else:
        print("Warning: No input filenames passed.")
        exit(-1)

    for index, input_filename in enumerate(filenames):
        print(f"File format conversion [{index + 1}]: {input_filename}")
        file_format(input_filename)
