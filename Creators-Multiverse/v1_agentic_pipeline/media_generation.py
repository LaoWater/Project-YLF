# media_generation.py
import asyncio
from typing import Literal, Optional
from visual_model import generate_and_save_image  # Import the visual model function

# from config import IMAGE_FILE_EXTENSION  # Assuming this is defined in your config

# Default configuration - adjust as needed
IMAGE_FILE_EXTENSION = "png"  # Can be moved to config.py


async def generate_visual_asset_for_platform(
        image_prompt: str,
        output_directory: str,  # e.g., "generated_posts/facebook"
        filename_base: str,  # e.g., "hello_world_intro_post"
        media_type: Literal["image"] = "image",  # Extend for "video" later
        model: str = "gpt-image-1",  # OpenAI model to use - # da-le3 feeling bit too anime/ai
        # Yet GPT-Image-1 Extraordinarily more slow than da-le3 - 15-20x slow - but in this kind of product - waiting time is our Friend.
        # And GPT-Image-1 - quickly becomes very expensive - averaging at 0.15$ per photo - (image quality high)
        # Must try google As well - Imagen 4
        file_extension: str = IMAGE_FILE_EXTENSION
) -> str:
    """
    Orchestrates the generation and saving of a visual asset using the visual_model.

    This function runs the synchronous generate_and_save_image in a separate thread
    to avoid blocking the asyncio event loop.

    Args:
        image_prompt: Text prompt for image generation.
        output_directory: Directory where the generated asset should be saved.
        filename_base: Base filename (without extension) for the generated asset.
        media_type: Type of media to generate (currently only "image" supported).
        model: OpenAI model to use for generation.
        file_extension: File extension for the generated image.

    Returns:
        The file path where the media was saved.

    Raises:
        NotImplementedError: If media_type is not "image".
        Exception: If image generation fails.
    """
    print(f"\n🖼️ Media Generation Task: Starting for {filename_base} in {output_directory}")
    print(f"🖼️ Media Generation Task: Using prompt: '{image_prompt}'")

    if media_type == "image":
        try:
            # Run the synchronous visual_model.generate_and_save_image in a thread
            # to avoid blocking the asyncio event loop
            file_path = await asyncio.to_thread(
                generate_and_save_image,
                image_prompt,
                output_directory,
                filename_base,
                file_extension,
                model
            )

            print(f"✅ Media Generation Task: Asset ready at {file_path}")
            return file_path

        except Exception as e:
            error_msg = f"Error during visual asset generation for {filename_base}: {e}"
            print(f"🚨 Media Generation Task: {error_msg}")
            # Re-raise the exception to make errors visible to the calling code
            raise Exception(error_msg)

    else:
        # Placeholder for video or other media types
        error_msg = f"Media type '{media_type}' not yet supported for generation."
        print(f"⚠️ Media Generation Task: {error_msg}")
        raise NotImplementedError(error_msg)


async def generate_multiple_visual_assets(
        prompts_and_configs: list[dict],
        base_output_directory: str = "generated_posts"
) -> list[str]:
    """
    Generate multiple visual assets concurrently.

    Args:
        prompts_and_configs: List of dictionaries containing:
            - prompt: Image generation prompt
            - platform: Platform name (e.g., "facebook", "instagram")
            - filename_base: Base filename
            - media_type: Type of media (default: "image")
        base_output_directory: Base directory for all generated assets

    Returns:
        List of file paths where assets were saved.
    """
    print(f"\n🎬 Batch Media Generation: Starting {len(prompts_and_configs)} tasks...")

    # Create tasks for concurrent execution
    tasks = []
    for config in prompts_and_configs:
        platform_dir = f"{base_output_directory}/{config.get('platform', 'default')}"

        task = generate_visual_asset_for_platform(
            image_prompt=config['prompt'],
            output_directory=platform_dir,
            filename_base=config['filename_base'],
            media_type=config.get('media_type', 'image')
        )
        tasks.append(task)

    try:
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and separate successful paths from errors
        successful_paths = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ Task {i + 1} failed: {result}")
            else:
                successful_paths.append(result)
                print(f"✅ Task {i + 1} completed: {result}")

        print(
            f"\n🎬 Batch Media Generation: Completed {len(successful_paths)}/{len(prompts_and_configs)} tasks successfully")
        return successful_paths

    except Exception as e:
        print(f"🚨 Batch Media Generation: Unexpected error: {e}")
        raise


# --- Demo/Test Functions ---
async def demo_single_generation():
    """Demo function for testing single image generation."""
    print("🧪 Demo: Single Image Generation")

    try:
        file_path = await generate_visual_asset_for_platform(
            image_prompt="A professional social media post background with modern gradient colors",
            output_directory="demo_output",
            filename_base="demo_post_bg"
        )
        print(f"Demo completed successfully: {file_path}")
        return file_path
    except Exception as e:
        print(f"Demo failed: {e}")
        return None


async def demo_batch_generation():
    """Demo function for testing batch image generation."""
    print("🧪 Demo: Batch Image Generation")

    configs = [
        {
            'prompt': 'A vibrant Instagram-style food photo with natural lighting',
            'platform': 'instagram',
            'filename_base': 'food_post_1'
        },
        {
            'prompt': 'A professional LinkedIn banner with business theme',
            'platform': 'linkedin',
            'filename_base': 'business_banner'
        },
        {
            'prompt': 'A fun Facebook cover photo with community vibes',
            'platform': 'facebook',
            'filename_base': 'community_cover'
        }
    ]

    try:
        results = await generate_multiple_visual_assets(configs, "demo_batch_output")
        print(f"Batch demo completed: {len(results)} assets generated")
        return results
    except Exception as e:
        print(f"Batch demo failed: {e}")
        return []


# --- Main execution for testing ---
async def main():
    """Main function for testing the media generation pipeline."""
    print("🚀 Media Generation Pipeline - Test Mode")

    # Test single generation
    await demo_single_generation()

    print("\n" + "=" * 50 + "\n")

    # Test batch generation
    await demo_batch_generation()

    print("\n🏁 All demos completed!")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())