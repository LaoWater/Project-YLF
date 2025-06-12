import os
import time
import base64
from openai import OpenAI

# --- Configuration ---
# Uses latest OpenAI GPT to generate photos for Platform Agents prompts
# Returned image is to be taken with Platform Agent Post text and saved.


client = OpenAI()


# --- Agent/Pipeline Step: Generate Image ---
def generate_and_save_image(prompt: str,
                            model: str = "gpt-image-1",  # or "dall-e-3"
                            n: int = 1,  # Number of images to generate
                            output_folder: str = ".",  # Save in the current folder
                            filename_prefix: str = "generated_img"
                            ) -> list[str]:  # Returns a list of saved file paths
    """
    Generates an image using OpenAI's DALL-E model and saves it locally.

    Args:
        prompt: The text prompt to generate the image from.

    Returns:
        A list of file paths where the images were saved.
        Returns an empty list if an error occurs.
    """
    saved_file_paths = []
    try:
        print(f"🎨 Generating image with GPT (model: {model}) for prompt: '{prompt}'...")


        model_params = {
            "model": model,
            "prompt": prompt
        }


        response = client.images.generate(**model_params)

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        for i, image_data in enumerate(response.data):
            if image_data.b64_json:
                image_bytes = base64.b64decode(image_data.b64_json)
                timestamp = int(time.time())
                # Sanitize prompt for filename (simple version)
                safe_prompt_suffix = "".join(c if c.isalnum() else "_" for c in prompt[:20]).strip("_")
                filename = f"{filename_prefix}_{safe_prompt_suffix}_{timestamp}_{i + 1}.png"
                file_path = os.path.join(output_folder, filename)

                with open(file_path, "wb") as f:
                    f.write(image_bytes)
                print(f"🖼️ Image saved successfully as: {file_path}")
                saved_file_paths.append(file_path)
            else:
                # This case should not happen if response_format is b64_json and successful
                print(f"⚠️ No b64_json data found for image {i + 1}.")
                if image_data.url:
                    print(f"   Image URL (if available): {image_data.url} (will not be saved locally by this function)")


    except Exception as e:
        print(f"❌ An error occurred during image generation or saving: {e}")
        # In a real pipeline, you might want to raise the exception
        # or return a specific error code/object.
        return []

    return saved_file_paths


# --- Simulate Pipeline Orchestration ---
if __name__ == "__main__":
    print("🚀 Starting Image Generation Pipeline Step...")

    # 1. Define the input for this step (e.g., from a previous agent or user input)
    image_prompt = input("Enter a prompt for the image you want to generate: ")
    if not image_prompt:
        image_prompt = "A futuristic cityscape at sunset with flying cars, photorealistic"
        print(f"No prompt entered, using default: '{image_prompt}'")

    # 2. Execute the image generation and saving step

    generated_files = generate_and_save_image(
        prompt=image_prompt,
        model="gpt-image-1",
        output_folder="generated_images"  # Save in a subfolder
    )

    if generated_files:
        print("\n✅ Image generation step completed.")
        print("Generated files:")
        for file_path in generated_files:
            print(f"  - {file_path}")

        print("\n下一步 (Next Step in Pipeline):")
        print("These files are now available locally.")
        print("The next agent in the orchestration pipeline could:")
        print("  1. Pick up these file(s) from the 'generated_images' folder.")
        print("  2. Perform further processing (e.g., resizing, watermarking).")
        print("  3. Upload them to a cloud storage (S3, Azure Blob, etc.).")
        print("  4. Send a notification or a reference (URL/path) to the web app.")
        print("     - If the web app and this script share a filesystem, the path might be enough.")
        print("     - If not, uploading to cloud storage and then providing a URL is common.")
    else:
        print("\n⚠️ Image generation step failed or produced no files.")

    print("\n🏁 Pipeline Step Finished.")
