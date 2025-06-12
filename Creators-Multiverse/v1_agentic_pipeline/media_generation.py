# media_generation.py
import asyncio
from typing import Literal
from visual_model import generate_and_save_image # Actual visual model
from config import IMAGE_FILE_EXTENSION

async def generate_visual_asset_for_platform(
    image_prompt: str,
    output_directory: str, # e.g., "generated_posts/facebook"
    filename_base: str,    # e.g., "hello_world_intro_post"
    media_type: Literal["image"] = "image" # Extend for "video" later
) -> str:
    """
    Orchestrates the generation and saving of a visual asset using the visual_model.
    Returns the path to the saved media file.
    Runs the synchronous generate_and_save_image in a separate thread.
    """
    print(f"\n🖼️ Media Generation Task: Starting for {filename_base} in {output_directory}")
    if media_type == "image":
        # Run the synchronous visual_model.generate_and_save_image in a thread
        # to avoid blocking the asyncio event loop.
        try:
            file_path = await asyncio.to_thread(
                generate_and_save_image,
                image_prompt,
                output_directory,
                filename_base,
                IMAGE_FILE_EXTENSION
            )
            print(f"🖼️ Media Generation Task: Asset ready at {file_path}")
            return file_path
        except Exception as e:
            print(f"🚨 Media Generation Task: Error during visual asset generation for {filename_base}: {e}")
            # Depending on desired behavior, you might return None, a placeholder, or re-raise
            raise # Re-raising for now to make errors visible
    else:
        # Placeholder for video or other media types
        print(f"⚠️ Media Generation Task: Media type '{media_type}' not yet supported for generation.")
        raise NotImplementedError(f"Media type '{media_type}' not supported.")