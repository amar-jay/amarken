import json

def convert_to_jsonl(input_filepath, output_filepath):
    # Read the raw, multi-line JSON text
    with open(input_filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    decoder = json.JSONDecoder()
    pos = 0
    
    with open(output_filepath, 'w', encoding='utf-8') as out_f:
        while pos < len(content):
            # Skip any whitespace or newlines between objects
            while pos < len(content) and content[pos].isspace():
                pos += 1
            if pos >= len(content):
                break
            
            # Decode the next JSON object from the text
            obj, new_pos = decoder.raw_decode(content, pos)
            
            # Write the object to the new file as a single line
            out_f.write(json.dumps(obj, ensure_ascii=False) + '\n')
            
            # Move the position forward
            pos = new_pos

    print(f"Successfully converted to {output_filepath}")

# Example usage:
# Make sure you have your original text saved in a file named 'input.txt'
if __name__ == "__main__":
    import sys

    input_filepath = sys.argv[1] if len(sys.argv) > 1 else None
    output_filepath = sys.argv[2] if len(sys.argv) > 2 else None

    if not input_filepath or not input_filepath.endswith(".jsonl"):
        print("Please provide the input file path as the first argument.")
        sys.exit(1)

    if not output_filepath or not output_filepath.endswith(".jsonl"):
        print("Please provide the output file path as the second argument.")
        sys.exit(1)

    convert_to_jsonl(input_filepath, output_filepath)