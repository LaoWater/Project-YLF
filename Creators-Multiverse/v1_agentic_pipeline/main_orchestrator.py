# main_orchestrator.py
import asyncio
import json
from typing import List, Optional, Dict
import os
from datetime import datetime, timezone


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
from cloud_storage_service import cloud_storage, upload_generated_post_files


async def generate_social_media_posts_pipeline(
        subject: str,
        target_platforms: List[str],
        requirements: Optional[Requirements] = None,
        posts_history: Optional[List[PostHistoryEntry]] = None,
        upload_to_cloud: bool = True  # New parameter to control cloud uploads
) -> Dict[str, any]:
    """
    Generate social media posts and optionally upload to cloud storage.

    Returns:
        Dictionary containing both local file paths and cloud storage information
    """
    print("🚀 Starting Social Media Post Generation Pipeline 🚀")

    # --- Generate Filename Base ---
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
        platform_adaptation_tasks.append(
            (platform_name, run_platform_adaptation_agent(platform_agent_input))
        )

    platform_agent_results_tuples = await asyncio.gather(*(task for _, task in platform_adaptation_tasks))

    platform_outputs_map: Dict[str, PlatformAgentOutput] = {}
    for i, platform_output in enumerate(platform_agent_results_tuples):
        platform_name = platform_adaptation_tasks[i][0]
        platform_outputs_map[platform_name] = platform_output

    # --- Media Generation (if applicable) ---
    final_posts_results: List[FinalGeneratedPost] = []
    media_generation_coroutines = []
    platforms_needing_media_info: List[tuple[str, str]] = []

    if decided_post_type in ["B", "C"]:  # Post types that require media
        for platform_name in target_platforms:
            platform_output = platform_outputs_map.get(platform_name)
            if not platform_output:
                print(f"⚠️ Warning: No platform output for {platform_name}, cannot determine media needs.")
                continue

            media_prompt = platform_output.get("platform_media_generation_prompt")
            if media_prompt:
                platform_dir = ensure_platform_folder_exists(BASE_OUTPUT_FOLDER, platform_name)
                media_generation_coroutines.append(
                    generate_visual_asset_for_platform(
                        image_prompt=media_prompt,
                        output_directory=platform_dir,
                        filename_base=filename_base,
                        media_type="image"
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
            print(f"🚨 Major error during media generation: {e}")
            generated_media_paths = [None] * len(platforms_needing_media_info)

    # --- Cloud Upload Tasks (Prepare) ---
    cloud_upload_tasks = []
    cloud_upload_results = []

    # --- Assemble Final Posts and Prepare Cloud Uploads ---
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
        media_file_path: Optional[str] = None

        # Handle media assets
        is_media_platform = any(p_name == platform_name for p_name, _ in platforms_needing_media_info)
        if is_media_platform:
            original_prompt_for_platform = next(
                (prompt for p_name, prompt in platforms_needing_media_info if p_name == platform_name), None)

            try:
                current_platform_media_index = [info[0] for info in platforms_needing_media_info].index(platform_name)
                if current_platform_media_index < len(generated_media_paths) and generated_media_paths[
                    current_platform_media_index]:
                    media_file_path = generated_media_paths[current_platform_media_index]
                    current_media_asset = SavedMediaAsset(
                        type="image",
                        file_path=media_file_path
                    )
                    media_prompt_used_for_this_platform = original_prompt_for_platform
            except (ValueError, IndexError) as e:
                print(f"🚨 Error mapping media to platform {platform_name}: {e}")

        # Prepare cloud upload task for this platform
        if upload_to_cloud:
            cloud_upload_task = upload_generated_post_files(
                filename_base=filename_base,
                platform=platform_name,
                text_content=text_content,
                media_file_path=media_file_path,
                media_generation_prompt=media_prompt_used_for_this_platform
            )
            cloud_upload_tasks.append((platform_name, cloud_upload_task))

        final_posts_results.append(FinalGeneratedPost(
            platform=platform_name,
            post_type=decided_post_type,
            text_file_path=text_file_path,
            media_asset=current_media_asset,
            original_text_content=text_content,
            media_generation_prompt_used=media_prompt_used_for_this_platform
        ))

    # --- Execute Cloud Uploads Concurrently ---
    if upload_to_cloud and cloud_upload_tasks:
        print(f"\n☁️ Starting {len(cloud_upload_tasks)} cloud upload tasks...")
        try:
            cloud_upload_results_tuples = await asyncio.gather(*(task for _, task in cloud_upload_tasks))

            # Map results back to platforms
            for i, upload_result in enumerate(cloud_upload_results_tuples):
                platform_name = cloud_upload_tasks[i][0]
                cloud_upload_results.append({
                    "platform": platform_name,
                    "upload_result": upload_result
                })

                # Update final_posts_results with cloud URLs
                for post in final_posts_results:
                    if post["platform"] == platform_name:
                        post["cloud_storage"] = upload_result
                        break

        except Exception as e:
            print(f"🚨 Error during cloud uploads: {e}")
            cloud_upload_results = []

    # --- Create and Upload Summary ---
    pipeline_summary = {
        "pipeline_id": f"{filename_base}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "subject": subject,
        "post_type": decided_post_type,
        "platforms": target_platforms,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "posts": final_posts_results,
        "cloud_uploads": cloud_upload_results,
        "requirements": requirements,
        "posts_history": posts_history
    }

    # Save summary locally
    summary_filename = os.path.join(BASE_OUTPUT_FOLDER, f"{filename_base}_summary.json")
    try:
        with open(summary_filename, "w", encoding='utf-8') as f:
            json.dump(pipeline_summary, f, indent=4, ensure_ascii=False)
        print(f"📋 Local summary saved to {summary_filename}")
    except IOError as e:
        print(f"🚨 Error saving local summary: {e}")

    # Upload summary to cloud
    if upload_to_cloud:
        try:
            summary_cloud_path = cloud_storage.generate_cloud_path(
                filename_base, "summary", "metadata", "json"
            )
            summary_upload_result = await cloud_storage.upload_json_data(
                pipeline_summary,
                summary_cloud_path,
                metadata={
                    "content_type": "pipeline_summary",
                    "subject": subject,
                    "post_type": decided_post_type,
                    "platforms": ",".join(target_platforms)
                }
            )
            pipeline_summary["summary_cloud_storage"] = summary_upload_result
            print(f"☁️ Summary uploaded to cloud: {summary_upload_result.get('public_url', 'URL not available')}")
        except Exception as e:
            print(f"🚨 Error uploading summary to cloud: {e}")

    print("\n✅ Social Media Post Generation Pipeline Complete! ✅")
    return pipeline_summary


async def main():
    """Main function with enhanced cloud integration."""

    sample_posts_history: List[PostHistoryEntry] = [
        {"post_type": "A", "count": 7, "score": 8},
        {"post_type": "B", "count": 5, "score": 9},
        {"post_type": "C", "count": 3, "score": 7}
    ]

    # Run the pipeline with cloud upload enabled
    pipeline_result = await generate_social_media_posts_pipeline(
        subject=DEFAULT_POST_SUBJECT,
        target_platforms=SUPPORTED_PLATFORMS,
        posts_history=sample_posts_history,
        upload_to_cloud=True  # Enable cloud uploads
    )

    print("\n" + "=" * 60)
    print("📊 PIPELINE EXECUTION SUMMARY")
    print("=" * 60)

    print(f"Pipeline ID: {pipeline_result['pipeline_id']}")
    print(f"Subject: {pipeline_result['subject']}")
    print(f"Post Type: {pipeline_result['post_type']}")
    print(f"Platforms: {', '.join(pipeline_result['platforms'])}")
    print(f"Generated At: {pipeline_result['generated_at']}")

    print(f"\n📱 GENERATED POSTS ({len(pipeline_result['posts'])} total):")
    for i, post_info in enumerate(pipeline_result['posts']):
        print(f"\n--- Post {i + 1}: {post_info['platform'].upper()} ---")
        print(f"Text Preview: {post_info['original_text_content'][:100]}...")

        if post_info.get('media_asset'):
            print(f"Media: {post_info['media_asset']['type']} - {post_info['media_asset']['file_path']}")

        # Show cloud storage info
        if post_info.get('cloud_storage'):
            cloud_info = post_info['cloud_storage']
            print(f"☁️ Cloud Storage:")
            for upload in cloud_info.get('uploads', []):
                if upload.get('success'):
                    print(f"  ✅ {upload.get('cloud_path', 'Unknown path')}")
                    print(f"     URL: {upload.get('public_url', 'URL not available')}")
                else:
                    print(f"  ❌ Upload failed: {upload.get('error', 'Unknown error')}")

    # Show summary cloud storage
    if pipeline_result.get('summary_cloud_storage'):
        summary_cloud = pipeline_result['summary_cloud_storage']
        if summary_cloud.get('success'):
            print(f"\n📋 Summary Cloud Storage:")
            print(f"  ✅ {summary_cloud.get('public_url', 'URL not available')}")
        else:
            print(f"  ❌ Summary upload failed: {summary_cloud.get('error', 'Unknown error')}")

    print(f"\n🌐 WEB APP INTEGRATION:")
    print(f"Your Vite/TSX web app can now retrieve all generated content from:")
    print(f"- Individual post files via their public URLs")
    print(f"- Complete pipeline summary via the summary JSON URL")
    print(f"- Use the pipeline_id '{pipeline_result['pipeline_id']}' to track this generation")

    print("\n🏁 All operations completed!")


if __name__ == "__main__":
    asyncio.run(main())