import json
import torch


def save_layer_weights(model_weights, layer_name, output_file_path):
    """
    Save the tensor values of a specified layer in the model weights to a file.

    Parameters:
    model_weights (dict): The model's weights.
    layer_name (str): The name of the layer to inspect.
    output_file_path (str): The path of the file to save the tensor values.
    """
    if layer_name in model_weights:
        # Convert the tensor to a list for easier inspection
        tensor_values = model_weights[layer_name].tolist()

        # Save the values to a file
        with open(output_file_path, 'w') as file:
            for row in tensor_values:
                file.write(' '.join(map(str, row)) + '\n')
        print(f"Tensor values for {layer_name} have been saved to {output_file_path}")
    else:
        print(f"{layer_name} not found in model weights")


def main():
    # Define the path to the params.json file
    params_file_path = r"C:\Users\baciu\Desktop\Neo Training\Transcendence\LLama3-1\llama-models\models\llama3_1\Meta-Llama-3.1-8B\params.json"

    print("Model params: \n")

    # Load and print the params.json file
    with open(params_file_path, "r") as file:
        params = json.load(file)
        print(json.dumps(params, indent=4))

    # Define the path to the consolidated.00.pth file
    model_weights_path = r"C:\Users\baciu\Desktop\Neo Training\Transcendence\LLama3-1\llama-models\models\llama3_1\Meta-Llama-3.1-8B\consolidated.00.pth"
    print("\n Model Weights: \n")

    # Load the model weights
    model_weights = torch.load(model_weights_path)

    # Print all layer names
    layer_names = model_weights.keys()
    for layer_name in layer_names:
        print(layer_name)

    # Define the layer names you want to inspect
    layers_to_inspect = [
        'layers.0.attention.wq.weight',
        'layers.0.attention.wk.weight',
        'layers.0.feed_forward.w1.weight',
        'layers.0.attention_norm.weight',
        'layers.0.ffn_norm.weight'
    ]

    # Save tensor values for each specified layer
    for layer_name in layers_to_inspect:
        output_file_path = f"{layer_name.replace('.', '_')}.txt"
        save_layer_weights(model_weights, layer_name, output_file_path)


if __name__ == "__main__":
    main()
