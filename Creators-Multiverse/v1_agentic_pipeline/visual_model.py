# visual_model.py
import os
import asyncio


# This is a synchronous function as per the initial thought process.
# If your actual visual model library is async, you can change this to `async def`.
# Otherwise, it will be run in a thread by media_generation.py.
def generate_and_save_image(image_prompt: str, output_directory: str, filename_base: str,
                            extension: str = "jpg") -> str:
    """
    Simulates generating an image based on a prompt and saves it.
    Returns the full path to the saved image.
    """
    print(f"\n🎨 Visual Model: Received prompt: '{image_prompt[:100]}...'")
    print(f"🎨 Visual Model: Simulating image generation for {filename_base}.{extension} in {output_directory}...")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    file_path = os.path.join(output_directory, f"{filename_base}.{extension}")

    # Simulate work and create a dummy file
    # In a real scenario, this would involve calling the actual image generation model
    # and saving the received image bytes to file_path.
    try:
        with open(file_path, "w") as f:
            f.write(f"Simulated image for prompt: {image_prompt}\n")
            f.write(f"This is a dummy {extension} file.\n")
        print(f"🎨 Visual Model: Successfully saved dummy image to {file_path}")
    except IOError as e:
        print(f"🚨 Visual Model: Error saving dummy image {file_path}: {e}")
        raise  # Or handle error appropriately

    # Simulate some delay for image generation
    # If this were a real blocking I/O call, it would be run in a thread.
    # For simulation, this synchronous sleep is fine. If it were a real CPU-bound task or
    # a long I/O operation, asyncio.to_thread would be used in the calling async function.
    # Since we are just creating a file, it's fast. A real model might take seconds/minutes.
    # time.sleep(2) # Simulating generation time - remove if using asyncio.to_thread for real model

    return file_path
