import os
import time
import base64
import asyncio
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()


def generate_and_save_image(image_prompt: str, output_directory: str, filename_base: str,
                            extension: str = "png", model: str = "gpt-image-1") -> str:
    """
    Generates an image using OpenAI's image generation model and saves it locally.
    This function is synchronous and should be called via asyncio.to_thread() for async contexts.

    Args:
        image_prompt: The text prompt to generate the image from
        output_directory: Directory where the image will be saved
        filename_base: Base name for the file (without extension)
        extension: File extension (default: png for OpenAI images)
        model: OpenAI model to use (default: gpt-image-1)

    Returns:
        str: Full path to the saved image file

    Raises:
        Exception: If image generation or saving fails
    """
    print(f"\n🎨 Visual Model: Received prompt: '{image_prompt[:100]}...'")
    print(f"🎨 Visual Model: Generating image using {model} for {filename_base}.{extension} in {output_directory}...")

    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    try:
        # Generate image using OpenAI
        print(f"🎨 Calling OpenAI image generation API...")

        model_params = {
            "model": model,
            "prompt": image_prompt,
        }

        response = client.images.generate(**model_params)

        # Process the first (and typically only) image from response
        if not response.data:
            raise Exception("No image data received from OpenAI API")

        image_data = response.data[0]

        if not image_data.b64_json:
            raise Exception("No base64 image data in API response")

        # Decode base64 image data
        image_bytes = base64.b64decode(image_data.b64_json)

        # Create timestamped filename to avoid conflicts
        timestamp = int(time.time())
        final_filename = f"{filename_base}_{timestamp}.{extension}"
        file_path = os.path.join(output_directory, final_filename)

        # Save image to file
        with open(file_path, "wb") as f:
            f.write(image_bytes)

        print(f"🎨 Visual Model: Successfully saved image to {file_path}")
        print(f"🎨 Image size: {len(image_bytes)} bytes")

        return file_path

    except Exception as e:
        error_msg = f"Visual Model: Error generating or saving image: {e}"
        print(f"🚨 {error_msg}")
        raise Exception(error_msg)


# Alternative async wrapper function (if you prefer to handle async at this level)
async def generate_and_save_image_async(image_prompt: str, output_directory: str, filename_base: str,
                                        extension: str = "png", model: str = "gpt-image-1") -> str:
    """
    Async wrapper for generate_and_save_image.
    Runs the synchronous OpenAI call in a thread to avoid blocking the event loop.

    Args:
        Same as generate_and_save_image

    Returns:
        str: Full path to the saved image file
    """
    print(f"🎨 Visual Model: Running image generation in thread to avoid blocking...")

    # Run the synchronous function in a thread
    return await asyncio.to_thread(
        generate_and_save_image,
        image_prompt,
        output_directory,
        filename_base,
        extension,
        model
    )


# Utility function for batch generation (if needed)
def generate_multiple_images(prompts: list[str], output_directory: str,
                             filename_prefix: str = "generated_img",
                             extension: str = "png", model: str = "gpt-image-1") -> list[str]:
    """
    Generate multiple images from a list of prompts.
    This is a synchronous function - use asyncio.to_thread() if calling from async context.

    Args:
        prompts: List of text prompts
        output_directory: Directory where images will be saved
        filename_prefix: Prefix for generated filenames
        extension: File extension
        model: OpenAI model to use

    Returns:
        list[str]: List of file paths for successfully generated images
    """
    saved_files = []

    for i, prompt in enumerate(prompts):
        try:
            filename_base = f"{filename_prefix}_{i + 1}"
            file_path = generate_and_save_image(
                prompt, output_directory, filename_base, extension, model
            )
            saved_files.append(file_path)
        except Exception as e:
            print(f"🚨 Failed to generate image {i + 1}/{len(prompts)}: {e}")
            # Continue with next image instead of stopping
            continue

    return saved_files


# Async version of batch generation
async def generate_multiple_images_async(prompts: list[str], output_directory: str,
                                         filename_prefix: str = "generated_img",
                                         extension: str = "png", model: str = "gpt-image-1") -> list[str]:
    """
    Async version of generate_multiple_images using concurrent execution.

    Args:
        Same as generate_multiple_images

    Returns:
        list[str]: List of file paths for successfully generated images
    """
    print(f"🎨 Visual Model: Starting async batch generation of {len(prompts)} images...")

    # Create tasks for concurrent execution
    tasks = []
    for i, prompt in enumerate(prompts):
        filename_base = f"{filename_prefix}_{i + 1}"
        task = generate_and_save_image_async(
            prompt, output_directory, filename_base, extension, model
        )
        tasks.append(task)

    # Execute all tasks concurrently and collect results
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and return successful file paths
    saved_files = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"🚨 Failed to generate image {i + 1}/{len(prompts)}: {result}")
        else:
            saved_files.append(result)

    print(f"🎨 Visual Model: Batch generation completed. {len(saved_files)}/{len(prompts)} images successful.")
    return saved_files


# Test function for standalone usage
if __name__ == "__main__":
    print("🚀 Testing Visual Model Integration...")

    # Test single image generation
    test_prompt = "A futuristic cityscape at sunset with flying cars, photorealistic"
    test_output_dir = "test_generated_images"
    test_filename = "test_image"

    try:
        result_path = generate_and_save_image(
            test_prompt,
            test_output_dir,
            test_filename
        )
        print(f"✅ Test successful! Image saved to: {result_path}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

    print("\n🏁 Visual Model Test Finished.")