import os
import json
from typing import Dict, List, TypedDict, Optional, Literal, Union
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
import asyncio  # For simulating async media generation

# --- Environment Variable Loading ---
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY_VALUE = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY_VALUE:
    raise ValueError("GEMINI_API_KEY environment variable is not set or not found in .env file.")

# --- Company & Request Configuration ---
COMPANY_NAME = "Terapie Acasa"
COMPANY_MISSION = "Connecting people with Therapists around the home in the comfort of their own home."
COMPANY_SENTIMENT = "Warm, Friendly, Inviting, Health-Oriented"
POST_SUBJECT = "We're excited to soon be opening our doors to the clients - are you ready? Make sure your profile is complete, accurate and verified for our algorithm to have best results."

# LLM Model Configuration
DECISION_LLM_MODEL = "gemini-2.5-flash-preview-05-20"
PLATFORM_LLM_MODEL = "gemini-2.5-flash-preview-05-20"


# --- Typed Dictionaries for Data Structures ---

class PostHistoryEntry(TypedDict):
    platform: str
    post_type: Literal["A", "B", "C", "D", "E", "F"]
    score: int
    text_summary: Optional[str]


class Requirements(TypedDict):
    min_length: Optional[int]
    max_length: Optional[int]
    must_include_keywords: Optional[List[str]]


class Layer2Input(TypedDict):
    company_name: str
    company_mission: str
    company_sentiment: str
    subject: str
    platforms_to_target: List[str]
    requirements: Optional[Requirements]
    posts_history: Optional[List[PostHistoryEntry]]


class Layer2Output(TypedDict):
    post_type: Literal["A", "B", "C"]
    core_post_text: str
    # media_generation_prompt: Optional[str] # Layer 2's initial idea for media, platform agent will refine/create its own
    # Let's simplify: Layer 2 only decides type and core text.
    # Platform agent is fully responsible for its media prompt.


class MediaAsset(TypedDict):
    type: Literal["image", "video"]
    url_or_description: str  # Placeholder for actual media URL or detailed description of generated asset


class PlatformAgentInput(TypedDict):
    company_name: str
    company_mission: str
    company_sentiment: str
    subject: str
    post_type_decision: Literal["A", "B", "C"]
    core_post_text_suggestion: str
    target_platform: str


class PlatformAgentOutput(TypedDict):
    platform_specific_text: str
    platform_media_generation_prompt: Optional[str]  # Null if post_type_decision is "A"


class FinalGeneratedPost(TypedDict):
    platform: str
    post_type: Literal["A", "B", "C"]  # The type decided by Layer 2
    text: str
    media_asset: Optional[MediaAsset]  # Contains type and URL/description of the media


# --- LLM Initialization ---
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

decision_llm = ChatGoogleGenerativeAI(
    google_api_key=GEMINI_API_KEY_VALUE,
    model=DECISION_LLM_MODEL,
    temperature=0.6,
    safety_settings=safety_settings,
    # convert_system_message_to_human=True # REMOVED as per user request to fix warning
)

platform_llm = ChatGoogleGenerativeAI(
    google_api_key=GEMINI_API_KEY_VALUE,
    model=PLATFORM_LLM_MODEL,
    temperature=0.7,
    safety_settings=safety_settings,
    # convert_system_message_to_human=True # REMOVED
)

# --- Layer 2: Decision Making LLM ---
LAYER_2_SYSTEM_PROMPT = """
You are a Master Social Media Strategist and Content Planner for {company_name}.
Your Mission: Analyze the provided information and decide on the optimal post structure (type and core text) for the given subject.

Company Mission: {company_mission}
Company Sentiment: {company_sentiment}
Target Platforms: {platforms_to_target}

Available Post Types:
- Type A: Text-only post.
- Type B: Text + Evocative Media (Photo/Video). Media enhances emotion/aesthetics.
- Type C: Text + Informative Media (Photo/Video with Captions/Graphics). Media explains/illustrates.

Your Task:
1.  Consider the `subject`, `company_mission`, `company_sentiment`, `target_platforms`.
2.  If `posts_history` is provided, analyze it for insights.
3.  If `requirements` are provided, try to adhere to them.
4.  Decide on the most appropriate `post_type` (A, B, or C).
5.  Write a `core_post_text`. This text is a foundational message to be adapted by platform-specific agents.

Output Format:
Return a single JSON object with the following keys:
- "post_type": string (must be "A", "B", or "C")
- "core_post_text": string
"""


async def run_layer_2_decision_maker(inputs: Layer2Input) -> Layer2Output:
    print("\n--- Running Layer 2: Decision Maker ---")
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=LAYER_2_SYSTEM_PROMPT.format(
            company_name=inputs["company_name"],
            company_mission=inputs["company_mission"],
            company_sentiment=inputs["company_sentiment"],
            platforms_to_target=", ".join(inputs["platforms_to_target"])
        )),
        HumanMessage(content=f"""
            Subject to address: {inputs['subject']}
            Specific requirements: {json.dumps(inputs['requirements']) if inputs['requirements'] else 'None'}
            Posts history: {json.dumps(inputs['posts_history']) if inputs['posts_history'] else 'No past history provided.'}

            Please provide your strategic decision in the specified JSON format.
        """)
    ])

    chain = prompt_template | decision_llm | JsonOutputParser()

    try:
        # The formatted SystemMessage already includes the necessary variables from inputs
        # The HumanMessage includes the dynamic parts.
        # No need to pass these to .ainvoke if they are part of the prompt template directly.
        response: Layer2Output = await chain.ainvoke({})  # Empty dict as all info is in the prompt
        print(
            f"Layer 2 Decision: Post Type: {response.get('post_type')}, Core Text: {response.get('core_post_text', '')[:100]}...")
        return response
    except Exception as e:
        print(f"Error in Layer 2: {e}")
        raise


# --- Media Generation (Refactored as per request) ---
# This function would ideally live in a separate file (e.g., media_generator.py)
async def generate_media_content(prompt: str, media_type: Literal["image", "video"]) -> MediaAsset:
    """
    Simulates generating media content (image or video) based on a prompt.
    In a real application, this would call services like Imagen or Veo.
    """
    print(f"\n--- Simulating Media Generation ({media_type.capitalize()}) ---")
    print(f"Received prompt: {prompt[:150]}...")

    await asyncio.sleep(1)  # Simulate API call delay

    # Simulate a returned URL or descriptive identifier
    extension = "mp4" if media_type == "video" else "jpg"
    generated_media_url = f"https://example.com/generated_media_{media_type}_{hash(prompt) % 1000}.{extension}"
    print(f"Simulated {media_type} asset URL: {generated_media_url}")

    return MediaAsset(type=media_type, url_or_description=generated_media_url)


# --- Layer 3: Platform Adaptation LLM ---
PLATFORM_AGENT_SYSTEM_PROMPT_TEMPLATE = """
You are an expert Social Media Content Creator for {company_name}, specializing in the {target_platform} platform.
Your goal is to adapt a core message into a compelling post tailored for {target_platform}, and if media is required, create a suitable generation prompt.

Company Mission: {company_mission}
Company Sentiment: {company_sentiment}
Original Subject: {subject}
Post Type Decided by Strategist: {post_type_decision} (A: Text-only, B: Text + Evocative Media, C: Text + Informative Media)
Core Post Text Suggestion from Strategist: {core_post_text_suggestion}

Your Tasks:
1.  Craft `platform_specific_text` based on the `core_post_text_suggestion`. This text MUST be perfectly tailored for {target_platform}, considering its typical audience, style, best practices (e.g., length, tone, use of emojis, hashtags, calls to action).
    - LinkedIn: Professional, insightful, slightly longer form, use 3-5 relevant professional hashtags.
    - Instagram: Visually-driven, engaging caption, can be shorter or medium length, use 5-10 relevant and popular hashtags, emojis are welcome.
    - (Adapt for other platforms like Twitter, Facebook as needed, following their best practices.)
2.  If `post_type_decision` is 'B' or 'C', you MUST generate a `platform_media_generation_prompt`.
    This prompt will be used to generate an {media_type_for_prompt} (image for Type B, video for Type C, unless context strongly suggests otherwise).
    It should be highly detailed and specific to the {target_platform}'s visual style AND the `platform_specific_text` you just wrote.
    - For Type B (Evocative): Focus on aesthetics, mood, brand feel.
    - For Type C (Informative): Focus on clarity, conveying information visually, potentially with text overlays or simple graphics for a video.
    Describe scene, style, colors, aspect ratio (e.g., square for Instagram post, 9:16 for Reels/Stories), mood, and any specific elements.

Output Format:
Return a single JSON object with the following keys:
- "platform_specific_text": string
- "platform_media_generation_prompt": string (null if post_type_decision is "A")
"""


async def run_platform_adaptation_agent(inputs: PlatformAgentInput) -> PlatformAgentOutput:
    print(f"\n--- Running Layer 3: Platform Adaptation for {inputs['target_platform']} ---")

    media_type_for_prompt = "image"  # Default for Type B
    if inputs['post_type_decision'] == 'C':
        media_type_for_prompt = "video"  # Default for Type C

    # Format the system prompt string with all necessary inputs
    formatted_system_prompt = PLATFORM_AGENT_SYSTEM_PROMPT_TEMPLATE.format(
        company_name=inputs["company_name"],
        company_mission=inputs["company_mission"],
        company_sentiment=inputs["company_sentiment"],
        subject=inputs["subject"],
        post_type_decision=inputs["post_type_decision"],
        core_post_text_suggestion=inputs["core_post_text_suggestion"],
        target_platform=inputs["target_platform"],
        media_type_for_prompt=media_type_for_prompt
    )

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=formatted_system_prompt),
        HumanMessage(
            content=f"Please generate the tailored content for {inputs['target_platform']}. Remember to output in the specified JSON format.")
    ])

    chain = prompt_template | platform_llm | JsonOutputParser()

    try:
        # All info is in the prompt template
        response: PlatformAgentOutput = await chain.ainvoke({})
        print(
            f"Platform Agent ({inputs['target_platform']}) Text: {response.get('platform_specific_text', '')[:100]}...")
        if response.get('platform_media_generation_prompt'):
            print(
                f"Platform Agent ({inputs['target_platform']}) Media Prompt: {response['platform_media_generation_prompt'][:100]}...")
        return response
    except Exception as e:
        print(f"Error in Platform Adaptation for {inputs['target_platform']}: {e}")
        raise


# --- Main Orchestration ---
async def generate_social_media_posts(
        subject: str,
        target_platforms: List[str],
        requirements: Optional[Requirements] = None,
        posts_history: Optional[List[PostHistoryEntry]] = None
) -> List[FinalGeneratedPost]:
    print("🚀 Starting Social Media Post Generation Pipeline 🚀")

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

    # Prepare tasks for platform adaptation
    platform_adaptation_tasks = []
    for platform_name in target_platforms:
        platform_agent_input = PlatformAgentInput(
            company_name=COMPANY_NAME,
            company_mission=COMPANY_MISSION,
            company_sentiment=COMPANY_SENTIMENT,
            subject=subject,
            post_type_decision=decided_post_type,
            core_post_text_suggestion=layer2_result["core_post_text"],
            target_platform=platform_name
        )
        platform_adaptation_tasks.append(
            (platform_name, run_platform_adaptation_agent(platform_agent_input))
        )

    # Run platform adaptations concurrently
    platform_agent_outputs_with_name = await asyncio.gather(*(task for _, task in platform_adaptation_tasks))

    # Store platform agent outputs indexed by platform name for easier lookup
    platform_outputs_map: Dict[str, PlatformAgentOutput] = {}
    for i, output in enumerate(platform_agent_outputs_with_name):
        platform_name = platform_adaptation_tasks[i][0]
        platform_outputs_map[platform_name] = output

    # Prepare tasks for media generation (if needed)
    media_generation_tasks = []
    platform_names_needing_media = []

    if decided_post_type in ["B", "C"]:
        for platform_name in target_platforms:
            platform_output = platform_outputs_map[platform_name]
            media_prompt = platform_output.get("platform_media_generation_prompt")
            if media_prompt:
                media_type_to_generate: Literal["image", "video"] = "image"  # Default for Type B
                if decided_post_type == "C":
                    media_type_to_generate = "video"  # Default for Type C

                media_generation_tasks.append(
                    generate_media_content(media_prompt, media_type_to_generate)
                )
                platform_names_needing_media.append(platform_name)
            else:  # Should not happen if type is B or C, but handle defensively
                print(
                    f"Warning: Platform agent for {platform_name} did not provide a media prompt for a Type {decided_post_type} post.")

    generated_media_assets: List[Optional[MediaAsset]] = []
    if media_generation_tasks:
        generated_media_assets = await asyncio.gather(*media_generation_tasks)

    # Assemble final posts
    final_posts: List[FinalGeneratedPost] = []
    media_asset_idx = 0
    for platform_name in target_platforms:
        platform_output = platform_outputs_map[platform_name]
        current_media_asset: Optional[MediaAsset] = None

        if decided_post_type in ["B", "C"] and platform_output.get("platform_media_generation_prompt"):
            if platform_name in platform_names_needing_media and media_asset_idx < len(generated_media_assets):
                current_media_asset = generated_media_assets[media_asset_idx]
                media_asset_idx += 1

        final_posts.append(FinalGeneratedPost(
            platform=platform_name,
            post_type=decided_post_type,
            text=platform_output["platform_specific_text"],
            media_asset=current_media_asset
        ))

    print("\n✅ Social Media Post Generation Pipeline Complete! ✅")
    return final_posts


# --- Example Usage ---
async def main_orchestration():  # Renamed to avoid conflict with asyncio.run(main())
    sample_posts_history: List[PostHistoryEntry] = [
        {"platform": "instagram", "post_type": "B", "score": 8,
         "text_summary": "Previous successful product launch image post."},
        {"platform": "linkedin", "post_type": "A", "score": 7,
         "text_summary": "Informative article about industry trends."},
    ]

    target_platforms = ["linkedin", "instagram"]

    generated_posts_list = await generate_social_media_posts(
        subject=POST_SUBJECT,
        target_platforms=target_platforms,
        requirements=None,
        posts_history=sample_posts_history
    )

    print("\n--- Individual Generated Posts (Summary) ---")
    for i, post in enumerate(generated_posts_list):
        print(f"\n--- Post {i + 1} for {post['platform']} (Type: {post['post_type']}) ---")
        print(f"Text: {post['text'][:150]}...")  # Print summary
        if post['media_asset']:
            print(f"Media Type: {post['media_asset']['type']}")
            print(f"Media URL/Desc: {post['media_asset']['url_or_description']}")
        else:
            print("Media: None")
        print("---")

    return generated_posts_list


if __name__ == "__main__":
    if not GEMINI_API_KEY_VALUE:
        print("Please set the GEMINI_API_KEY in your .env file.")
    else:
        # Run the main orchestration
        final_generated_posts = asyncio.run(main_orchestration())

        # Output the final "beautiful JSON"
        final_json_output = json.dumps(final_generated_posts, indent=4)
        print("\n\n--- Final JSON Output ---")
        print(final_json_output)

        # Optionally, save to a file
        output_filename = "social_media_posts_output.json"
        with open(output_filename, "w") as f:
            f.write(final_json_output)
        print(f"\n✅ Final JSON output saved to {output_filename}")