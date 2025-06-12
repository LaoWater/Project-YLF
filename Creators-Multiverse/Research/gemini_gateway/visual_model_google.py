import os
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image
# from google.cloud.aiplatform import transport # Usually not needed explicitly unless troubleshooting
import datetime
import asyncio # <<<< ADDED IMPORT FOR ASYNCIO
from typing import Optional, Literal # For TypedDict if MediaAsset is defined here

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

if not GCP_PROJECT_ID:
    raise ValueError("GCP_PROJECT_ID environment variable is not set.")

# --- TypedDict for MediaAsset (if not imported from elsewhere) ---
# If MediaAsset is defined in your main script, you'd import it.
# For standalone use of this snippet, define it:
class MediaAsset(dict): # Using dict for simplicity if TypedDict is not critical here
    type: Literal["image", "video"]
    url_or_description: str

# --- Vertex AI Initialization ---
_vertex_ai_initialized = False # Flag to ensure it's initialized only once

def initialize_vertex_ai(project_id: str, location: str):
    """Initializes the Vertex AI SDK if not already initialized."""
    global _vertex_ai_initialized
    if _vertex_ai_initialized:
        # print("Vertex AI already initialized.")
        return
    try:
        vertexai.init(project=project_id, location=location)
        _vertex_ai_initialized = True
        print(f"Vertex AI initialized for project '{project_id}' in location '{location}'.")
    except Exception as e:
        print(f"Error initializing Vertex AI: {e}")
        raise

# --- Image Generation Function ---
async def generate_image_with_imagen( # This function itself doesn't need to be async
    prompt: str,                     # as the SDK call model.generate_images is synchronous.
    output_folder: str = "generated_media", # However, making it async allows it to be awaited
    model_name: str = "imagegeneration@006" # if called from an async orchestrator.
) -> Optional[str]:
    """
    Generates an image using Imagen on Vertex AI and saves it locally.
    The underlying SDK call is synchronous.
    """
    print(f"\n--- Generating Image with Imagen ---")
    print(f"Prompt: {prompt[:150]}...")

    # Ensure Vertex AI is initialized
    # This should be called once at the application start.
    # If called per function, ensure it's idempotent or use a flag.
    if not _vertex_ai_initialized:
        print("Warning: Vertex AI not explicitly initialized before calling generate_image_with_imagen.")
        print("Attempting to initialize now. Consider initializing once at script start.")
        try:
            initialize_vertex_ai(GCP_PROJECT_ID, GCP_LOCATION)
        except Exception as e:
            print(f"Failed to auto-initialize Vertex AI: {e}")
            return None


    try:
        model = ImageGenerationModel.from_pretrained(model_name)
        response = model.generate_images(
            prompt=prompt,
            number_of_images=1,
        )

        if not response.images:
            print("Image generation failed: No images returned in the response.")
            return None

        generated_image: Image = response.images[0]
        os.makedirs(output_folder, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize prompt for filename (simple example)
        safe_prompt_suffix = "".join(c for c in prompt if c.isalnum())[:20]
        filename = f"imagen_{timestamp}_{safe_prompt_suffix}.png"
        filepath = os.path.join(output_folder, filename)

        generated_image.save(location=filepath, include_generation_parameters=False)
        print(f"Image successfully generated and saved to: {filepath}")
        return filepath

    except Exception as e:
        print(f"An error occurred during image generation or saving: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Media Content Generation (Actual Service) ---
async def generate_media_content_actual(prompt: str, media_type: Literal["image", "video"]) -> Optional[MediaAsset]:
    """
    Generates media content using actual Google Cloud services.
    Currently supports image generation with Imagen.
    """
    if media_type == "image":
        # The generate_image_with_imagen call is synchronous,
        # but we run it in the default executor if we want to be super careful
        # about not blocking the event loop for too long, though for single image
        # generation, it might be acceptable directly.
        # For simplicity and because the core SDK call is blocking:
        # saved_image_path = await generate_image_with_imagen(prompt=prompt)

        # To truly run the blocking SDK call in a separate thread and await it:
        loop = asyncio.get_event_loop()
        saved_image_path = await loop.run_in_executor(
            None,  # Uses the default ThreadPoolExecutor
            sync_generate_image_with_imagen_wrapper, # Wrapper for the sync function
            prompt
        )

        if saved_image_path:
            return MediaAsset(type="image", url_or_description=saved_image_path)
        else:
            print(f"Failed to generate image for prompt: {prompt}")
            return None
    elif media_type == "video":
        print(f"Video generation for prompt '{prompt}' is not yet implemented with a real service.")
        await asyncio.sleep(0.5) # Simulate API call delay
        return MediaAsset(type="video", url_or_description=f"simulated_video_for_prompt_{hash(prompt) % 1000}.mp4")
    else:
        print(f"Unsupported media type: {media_type}")
        return None

def sync_generate_image_with_imagen_wrapper(prompt: str) -> Optional[str]:
    """
    Synchronous wrapper for generate_image_with_imagen to be used with run_in_executor.
    This function should not be async itself.
    """
    # Since generate_image_with_imagen was made async only to be awaitable,
    # and its core logic (SDK call) is synchronous, we can call it directly
    # if we make a synchronous version of it, or call a synchronous core part.

    # Let's make a synchronous version of the core logic for clarity here.
    print(f"\n--- (Sync Wrapper) Generating Image with Imagen ---")
    print(f"Prompt: {prompt[:150]}...")

    if not _vertex_ai_initialized:
        print("Warning: Vertex AI not explicitly initialized before calling generate_image_with_imagen.")
        # In a real app, initialization should happen reliably at startup.
        # Forcing an attempt here for robustness in standalone tests:
        try:
            initialize_vertex_ai(GCP_PROJECT_ID, GCP_LOCATION)
        except Exception as e:
            print(f"Failed to auto-initialize Vertex AI: {e}")
            return None # Cannot proceed without initialization

    model_name = "imagegeneration@006" # Or from config
    output_folder = "generated_media" # Or from config

    try:
        model = ImageGenerationModel.from_pretrained(model_name)
        response = model.generate_images(
            prompt=prompt,
            number_of_images=1,
        )

        if not response.images:
            print("Image generation failed: No images returned in the response.")
            return None

        generated_image: Image = response.images[0]
        os.makedirs(output_folder, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt_suffix = "".join(c for c in prompt if c.isalnum())[:20]
        filename = f"imagen_{timestamp}_{safe_prompt_suffix}.png"
        filepath = os.path.join(output_folder, filename)

        generated_image.save(location=filepath, include_generation_parameters=False)
        print(f"Image successfully generated and saved to: {filepath}")
        return filepath

    except Exception as e:
        print(f"An error occurred during image generation or saving: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Example Usage (for standalone testing of this file) ---
async def main_example_usage():
    """Example of calling the image generation function."""
    if not GCP_PROJECT_ID:
        print("GCP_PROJECT_ID is not set. Cannot run example.")
        return

    try:
        initialize_vertex_ai(GCP_PROJECT_ID, GCP_LOCATION) # Initialize once
    except Exception as e:
        print(f"Failed to initialize Vertex AI for example usage: {e}")
        return

    test_prompt = "A vibrant coral reef teeming with colorful fish and a hidden treasure chest, digital art style."

    print("\n--- Testing with generate_media_content_actual (Image) ---")
    image_asset = await generate_media_content_actual(prompt=test_prompt, media_type="image")
    if image_asset:
        print(f"Generated image asset: {image_asset}")
    else:
        print("Image generation via generate_media_content_actual failed.")

    print("\n--- Testing with generate_media_content_actual (Video Simulation) ---")
    video_asset = await generate_media_content_actual(prompt="A friendly robot waving hello", media_type="video")
    if video_asset:
        print(f"Simulated video asset: {video_asset}")


if __name__ == "__main__":
    # This script can be run standalone to test image generation
    # Ensure your environment variables (GCP_PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS) are set.
    if not GCP_PROJECT_ID:
        print("Please set the GCP_PROJECT_ID environment variable to run this example.")
        print("e.g., export GCP_PROJECT_ID='your-gcp-project-id'")
    else:
        print("Running standalone example for image generation...")
        asyncio.run(main_example_usage())