import json

# File paths
input_file_path = 'Data/v4_step5_creating_prompt_content_pairs.jsonl'
output_file_path = 'Data/v4_step6_formatted_for_OpenAI_finetuning.jsonl'

# System message content
system_message_content = "Lao is a Healer and a Humble Student of Life with the purpose to listen, understand and advise based on his knowledge."

# Open the input and output files
with open(input_file_path, 'r', encoding='utf-8') as infile, \
     open(output_file_path, 'w', encoding='utf-8') as outfile:

    for line_num, line in enumerate(infile, start=1):
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        try:
            # Parse the input JSON line
            data = json.loads(line)

            prompt = data.get('prompt', '').strip()
            content = data.get('content', '').strip()

            if not prompt or not content:
                print(f"Skipping line {line_num}: Missing 'prompt' or 'content'")
                continue

            # Create the new formatted entry
            formatted_entry = {
                "messages": [
                    {"role": "system", "content": system_message_content},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": content}
                ]
            }

            # Write the formatted entry to the output file
            outfile.write(json.dumps(formatted_entry) + '\n')

        except json.JSONDecodeError as e:
            print(f"Error parsing line {line_num}: {e}")
            continue

print(f"Data has been transformed and saved to {output_file_path}")
