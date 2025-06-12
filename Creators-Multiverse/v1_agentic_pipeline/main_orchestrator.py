# main_orchestrator.py
import asyncio
import json
from typing import List, Optional, Dict
import os

from config import (
    COMPANY_NAME, COMPANY_MISSION, COMPANY_SENTIMENT, DEFAULT_POST_SUBJECT,
    SUPPORTED_PLATFORMS, BASE_OUTPUT_FOLDER
)
from data_models import (
    Layer2Input, PostHistoryEntry, Requirements,
    PlatformAgentInput, PlatformAgentOutput,
    FinalGeneratedPost, SavedMediaAsset
)
from llm_services import run_layer_2_decision_maker, run_platform_adaptation_agent
from media_generation import generate_visual_asset_for_platform
from file_utils import get_filename_base, ensure_platform_folder_exists, save_text_content


async def generate_social_media_posts_pipeline(
        subject: str,
        target_platforms: List[str],
        requirements: Optional[Requirements] = None,
        posts_history: Optional[List[PostHistoryEntry]] = None
) -> List[FinalGeneratedPost]:
    print("🚀 Starting Social Media Post Generation Pipeline 🚀")

    # --- Generate Filename Base ---
    # This filename base will be used for all artifacts related to this subject
    filename_base = get_filename_base(subject)
    print(f"🎬 Using filename base: {filename_base}")

    # --- Layer 2: Strategic Decision ---
    layer2_input_data = Layer2Input(
        company_name=COMPANY_NAME,
        company_mission=COMPANY_MISSION,
        company_sentiment=COMPANY_SENTIMENT,
        subject=subject,
        platforms_to_target=target_platforms,
        requirements=requirements,
        posts_history=posts_history
    )
    layer2_result = await run_layer_2_decision_maker(layer2_input_data)
    decided_post_type = layer2_result["post_type"]
    core_post_text = layer2_result["core_post_text"]

    # --- Layer 3: Platform Adaptation ---
    platform_adaptation_tasks = []
    for platform_name in target_platforms:
        platform_agent_input = PlatformAgentInput(
            company_name=COMPANY_NAME,
            company_mission=COMPANY_MISSION,
            company_sentiment=COMPANY_SENTIMENT,
            subject=subject,
            post_type_decision=decided_post_type,
            core_post_text_suggestion=core_post_text,
            target_platform=platform_name
        )
        # Store tuple of (platform_name, task)
        platform_adaptation_tasks.append(
            (platform_name, run_platform_adaptation_agent(platform_agent_input))
        )

    # Gather results, maintaining association with platform_name
    platform_agent_results_tuples = await asyncio.gather(*(task for _, task in platform_adaptation_tasks))

    platform_outputs_map: Dict[str, PlatformAgentOutput] = {}
    for i, platform_output in enumerate(platform_agent_results_tuples):
        platform_name = platform_adaptation_tasks[i][0]  # Get platform_name from the original list
        platform_outputs_map[platform_name] = platform_output

    # --- Media Generation (if applicable) & File Saving ---
    final_posts_results: List[FinalGeneratedPost] = []
    media_generation_coroutines = []  # list of coroutines to run
    # Store tuples of (platform_name, image_prompt) for tasks that need media
    platforms_needing_media_info: List[tuple[str, str]] = []

    if decided_post_type in ["B", "C"]:  # Post types that require media
        for platform_name in target_platforms:
            platform_output = platform_outputs_map.get(platform_name)
            if not platform_output:
                print(f"⚠️ Warning: No platform output for {platform_name}, cannot determine media needs.")
                continue

            media_prompt = platform_output.get("platform_media_generation_prompt")
            if media_prompt:  # If the platform agent provided a prompt
                platform_dir = ensure_platform_folder_exists(BASE_OUTPUT_FOLDER, platform_name)
                # Add coroutine and info to lists
                media_generation_coroutines.append(
                    generate_visual_asset_for_platform(
                        image_prompt=media_prompt,
                        output_directory=platform_dir,
                        filename_base=filename_base,
                        media_type="image"  # Hardcoded to image for now
                    )
                )
                platforms_needing_media_info.append((platform_name, media_prompt))
            else:
                print(
                    f"ℹ️ Info: Platform agent for {platform_name} did not request media for post type {decided_post_type}.")

    # Run all media generation tasks concurrently
    generated_media_paths: List[str] = []
    if media_generation_coroutines:
        print(f"\n⏳ Starting {len(media_generation_coroutines)} media generation tasks...")
        try:
            generated_media_paths = await asyncio.gather(*media_generation_coroutines)
        except Exception as e:
            print(f"🚨🚨 Major error during asyncio.gather for media generation: {e}")
            # Handle this case, e.g., by setting paths to None or using placeholders
            # For now, if one fails, gather might stop others or return exceptions.
            # Fill with None for failed tasks to match length of platforms_needing_media_info
            generated_media_paths = [None] * len(platforms_needing_media_info)  # Simplified error handling

    # --- Assemble Final Posts and Save Text ---
    media_asset_idx = 0
    for platform_name in target_platforms:
        platform_output = platform_outputs_map.get(platform_name)
        if not platform_output:
            print(f"⚠️ Warning: No platform output for {platform_name} during final assembly. Skipping.")
            continue

        platform_dir = ensure_platform_folder_exists(BASE_OUTPUT_FOLDER, platform_name)
        text_content = platform_output["platform_specific_text"]
        text_file_path = save_text_content(platform_dir, filename_base, text_content)

        current_media_asset: Optional[SavedMediaAsset] = None
        media_prompt_used_for_this_platform: Optional[str] = None

        # Check if this platform was one that requested media
        # and has a corresponding successfully generated media path
        is_media_platform = any(p_name == platform_name for p_name, _ in platforms_needing_media_info)

        if is_media_platform:
            # Find the original prompt for this platform
            original_prompt_for_platform = next(
                (prompt for p_name, prompt in platforms_needing_media_info if p_name == platform_name), None)

            if media_asset_idx < len(generated_media_paths) and generated_media_paths[media_asset_idx]:
                # This platform *should* have media generated for it.
                # Find its corresponding path from generated_media_paths
                # This assumes order is preserved from media_generation_coroutines to generated_media_paths

                # A more robust way: map paths back if needed, but simple indexing works if gather preserves order
                # and no tasks were skipped *before* calling gather.
                # We need to find which media path belongs to which platform.
                # Let's find the correct index based on platforms_needing_media_info

                # Find the index of this platform in the list of platforms that needed media
                try:
                    current_platform_media_index = [info[0] for info in platforms_needing_media_info].index(
                        platform_name)
                    if current_platform_media_index < len(generated_media_paths) and generated_media_paths[
                        current_platform_media_index]:
                        media_file_path = generated_media_paths[current_platform_media_index]
                        current_media_asset = SavedMediaAsset(
                            type="image",  # Hardcoded for now
                            file_path=media_file_path
                        )
                        media_prompt_used_for_this_platform = original_prompt_for_platform
                except ValueError:
                    # This platform was not in platforms_needing_media_info, so no media for it.
                    pass
                except IndexError:
                    print(f"🚨 Error: Index out of bounds when trying to fetch media path for {platform_name}.")

            # This simple increment assumes that platforms_needing_media_info maps 1:1 with generated_media_paths
            # This is true if asyncio.gather preserves order and all tasks were attempted.
            # media_asset_idx += 1 # This logic is tricky, revised above.

        final_posts_results.append(FinalGeneratedPost(
            platform=platform_name,
            post_type=decided_post_type,
            text_file_path=text_file_path,
            media_asset=current_media_asset,
            original_text_content=text_content,  # Adding original text for easier review
            media_generation_prompt_used=media_prompt_used_for_this_platform
        ))

    print("\n✅ Social Media Post Generation Pipeline Complete! ✅")
    return final_posts_results


async def main():
    # Ensure OPENAI_API_KEY is loaded (already checked in config.py)
    # from config import OPENAI_API_KEY_VALUE # Not needed here, checked on import

    sample_posts_history: List[PostHistoryEntry] = [
        {"post_type": "A", "count": 7, "score": 8},
        {"post_type": "B", "count": 5, "score": 9},
        {"post_type": "C", "count": 3, "score": 7}
    ]

    # example_requirements: Requirements = {
    #     "min_length": 50,
    #     "max_length": 200,
    #     "must_include_keywords": ["empowering creators", COMPANY_NAME]
    # }

    final_generated_outputs = await generate_social_media_posts_pipeline(
        subject=DEFAULT_POST_SUBJECT,
        target_platforms=SUPPORTED_PLATFORMS,
        # requirements=example_requirements,
        posts_history=sample_posts_history
    )

    print("\n--- Summary of Generated Posts ---")
    for i, post_info in enumerate(final_generated_outputs):
        print(f"\n--- Output {i + 1} for {post_info['platform']} (Type: {post_info['post_type']}) ---")
        print(f"Text File: {post_info['text_file_path']}")
        print(f"Text Content (preview): {post_info['original_text_content'][:150]}...")
        if post_info['media_asset']:
            print(f"Media Type: {post_info['media_asset']['type']}")
            print(f"Media File: {post_info['media_asset']['file_path']}")
            if post_info['media_generation_prompt_used']:
                print(f"Media Prompt Used: {post_info['media_generation_prompt_used'][:100]}...")
        else:
            print("Media: None generated or requested for this platform.")
        print("---")

    # Save a summary JSON of all generated posts metadata
    summary_filename = os.path.join(BASE_OUTPUT_FOLDER, "all_posts_summary.json")
    try:
        with open(summary_filename, "w", encoding='utf-8') as f:
            json.dump(final_generated_outputs, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Overall summary saved to {summary_filename}")
    except IOError as e:
        print(f"🚨 Error saving summary JSON {summary_filename}: {e}")


if __name__ == "__main__":
    # The API key check is implicitly done when config.py is imported.
    # If it fails, an error is raised, and the script won't proceed.
    asyncio.run(main())