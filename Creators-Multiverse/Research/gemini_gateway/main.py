import os
import json
from typing import Dict, List, TypedDict, Optional, Literal, Union
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
import asyncio

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
DECISION_LLM_MODEL = "gemini-2.5-flash-preview-05-20"  # Changed to flash for potentially faster L2
PLATFORM_LLM_MODEL = "gemini-2.5-flash-preview-05-20"  # Kept Pro for nuanced platform adaptation


# --- Typed Dictionaries for Data Structures ---

class PostHistoryEntry(TypedDict):
    post_type: Literal["A", "B", "C"]
    count: int
    score: int



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


class MediaAsset(TypedDict):
    type: Literal["image", "video"]
    url_or_description: str


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
    platform_media_generation_prompt: Optional[str]


class FinalGeneratedPost(TypedDict):
    platform: str
    post_type: Literal["A", "B", "C"]
    text: str
    media_asset: Optional[MediaAsset]


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
    temperature=0.5,
    safety_settings=safety_settings,
)

platform_llm = ChatGoogleGenerativeAI(
    google_api_key=GEMINI_API_KEY_VALUE,
    model=PLATFORM_LLM_MODEL,
    temperature=0.7,  # Slightly higher for more creative platform adaptation
    safety_settings=safety_settings,
)

# --- Layer 2: Decision Making LLM ---
# *Later add Type D & E (Video)
LAYER_2_SYSTEM_PROMPT = """
You are a Master Social Media Strategist and Content Planner for {company_name}.
Your Mission: Analyze the provided information and decide on the optimal post structure (type and core text) for the given subject.

Company Mission: {company_mission}
Company Sentiment: {company_sentiment}
Target Platforms: {platforms_to_target}

Available Post Types:
- Type A: Text-only post.
- Type B: Text + Evocative Media (Photo). Media enhances emotion/aesthetics.
- Type C: Text + Informative Media (Photo with Captions & Background Graphics). Media explains/illustrates.

Your Task:
1.  Consider the `subject`, `company_mission`, `company_sentiment`, `target_platforms`.
2.  If `posts_history` is provided, analyze it for insights, trying to balance out the post types and once the count is over 20, can also be influenced by scores.
3.  If `requirements` are provided, try to adhere to them.
4.  Decide on the most appropriate `post_type` (A, B, or C).
5.  Write a `core_post_text`. This text is a foundational message to be passed on and adapted by platform-specific Agents.

Output Format:
Return a single JSON object with the following keys:
- "post_type": string (must be "A", "B", or "C")
- "core_post_text": string
"""


async def run_layer_2_decision_maker(inputs: Layer2Input) -> Layer2Output:
    print("\n--- Running Layer 2: Decision Maker ---")

    # Construct the prompt messages
    system_message_content = LAYER_2_SYSTEM_PROMPT.format(
        company_name=inputs["company_name"],
        company_mission=inputs["company_mission"],
        company_sentiment=inputs["company_sentiment"],
        platforms_to_target=", ".join(inputs["platforms_to_target"])
    )
    human_message_content = f"""
            Subject to address: {inputs['subject']}
            Specific requirements: {json.dumps(inputs['requirements']) if inputs['requirements'] else 'None'}
            Posts history: {json.dumps(inputs['posts_history']) if inputs['posts_history'] else 'No past history provided.'}

            Please provide your strategic decision in the specified JSON format.
        """

    prompt_messages_list = [
        SystemMessage(content=system_message_content),
        HumanMessage(content=human_message_content)
    ]
    prompt_template = ChatPromptTemplate.from_messages(prompt_messages_list)

    # --- Print the full prompt being sent to the model ---
    print("\n--- Layer 2: Full Prompt to LLM ---")
    for message in prompt_messages_list:
        print(f"--- Message Type: {message.type.upper()} ---")
        print(message.content)
        print("--- End Message ---")
    print("--- End of Full Prompt ---")
    # --- End of prompt print ---

    # Define the chain up to the LLM (without the final JSON parser yet)
    chain_upto_llm = prompt_template | decision_llm
    json_parser = JsonOutputParser()

    raw_llm_output_str = ""  # Initialize in case of early error

    try:
        # Get the raw response from the LLM
        llm_response_message = await chain_upto_llm.ainvoke({})
        raw_llm_output_str = llm_response_message.content

        # --- Print the raw LLM response ---
        print("\n--- Layer 2: Raw LLM Response ---")
        print(raw_llm_output_str)
        print("--- End of Raw LLM Response ---")
        # --- End of raw LLM response print ---

        # Now, parse the raw string response
        response: Layer2Output = json_parser.parse(raw_llm_output_str)

        print(
            f"\nLayer 2 Decision (Parsed): Post Type: {response.get('post_type')}, Core Text: {response.get('core_post_text', '')[:100]}...")
        return response
    except Exception as e:
        print(f"Error in Layer 2: {e}")
        if raw_llm_output_str: # If we got a response before the error (e.g., parsing error)
            print(f"Problematic raw LLM output on error:\n{raw_llm_output_str}")
        raise


# --- Media Generation ---
async def generate_media_content(prompt: str, media_type: Literal["image", "video"]) -> MediaAsset:
    print(f"\n--- Simulating Media Generation ({media_type.capitalize()}) ---")
    print(f"Received prompt: {prompt[:150]}...")
    await asyncio.sleep(0.5)  # Reduced sleep time for faster testing
    extension = "mp4" if media_type == "video" else "jpg"
    generated_media_url = f"https://example.com/generated_media_{media_type}_{hash(prompt) % 1000}.{extension}"
    print(f"Simulated {media_type} asset URL: {generated_media_url}")
    return MediaAsset(type=media_type, url_or_description=generated_media_url)


# --- Layer 3: Platform Adaptation LLM ---

# --- Specific System Prompts for Each Platform ---
LINKEDIN_SYSTEM_PROMPT = """
You are an expert Social Media Content Creator for {company_name}, specializing in LinkedIn.
Your goal is to adapt a core message into a professional and engaging LinkedIn post.

Company Name: {company_name}
Company Mission: {company_mission}
Company Sentiment: {company_sentiment}
Original Subject: {subject}
Post Type Decided by Strategist: {post_type_decision} (A: Text-only, B: Text + Evocative Photo, C: Text + Informative Photo)
You should take into consideration the recommandation, but not rigidly attach to it.
In some cases, you are free to change the type of post - from B to A if more appropriate for your platform specific.
Core Post Text Suggestion from Strategist: {core_post_text_suggestion}

LinkedIn Specific Guidelines:
-   Tone: Professional, insightful, authoritative, and value-driven. Align with "{company_sentiment}".
-   Length: Short to Medium to long-form is acceptable, depending on context.
-   Structure: Use clear paragraphs. Bullet points or numbered lists can enhance readability for complex information. Start with a strong hook related to "{subject}".
-   Call to Action: Encourage professional engagement (e.g., "What are your thoughts on this?", "Learn more about completing your profile by...", "Visit our {company_name} page to ensure you're ready.").
-   Hashtags: Use 3-5 relevant, professional hashtags. Consider including a brand hashtag if applicable, and terms related to therapy, well-being, and {company_name}'s services.
-   Media (if Type B/C):
    -   Type B (Evocative): Professional, high-quality images that reflect the trust and care of {company_name}. Could be abstract representations of comfort, connection, or growth.
    -   Type C (Informative): Visually engaging but calm background + captions with key messages from "{subject}". Text overlays should be stylish and readable.
    -   Media Prompt: If generating `platform_media_generation_prompt` for an {media_type_for_prompt}, 
    Craft a prompt to be fed into a Visual Generation Model. Aspect ratio for linkedin 16:9.

Your Tasks:
1.  Craft `platform_specific_text` adhering to the above LinkedIn guidelines.
2.  If `post_type_decision` is 'B' or 'C', generate a detailed `platform_media_generation_prompt` for an {media_type_for_prompt}.

Output Format:
Return a single JSON object with the keys "platform_specific_text" and "platform_media_generation_prompt" (null if Type A).
"""

INSTAGRAM_SYSTEM_PROMPT = """
You are a creative Social Media Content Creator for {company_name}, specializing in Instagram.
Your goal is to adapt a core message into a visually appealing and engaging Instagram post.

Company Name: {company_name}
Company Mission: {company_mission}
Company Sentiment: {company_sentiment}
Original Subject: {subject}
Post Type Decided by Strategist: {post_type_decision} (A: Text-only, B: Text + Evocative Photo, C: Text + Informative Photo)
Core Post Text Suggestion from Strategist: {core_post_text_suggestion}

Instagram Specific Guidelines:
-   Tone: Engaging, friendly, authentic, and visually descriptive. "{company_sentiment}" should shine through.
-   Caption Length: Short to medium. Break up long text with emojis or line breaks. The caption must complement the visual.
-   Engagement: Use questions related to readiness, well-being, or the comfort of home therapy. Encourage interaction (likes, comments, shares, saves).
-   Emojis: Use relevant emojis (e.g., 🏡, ✨, ❤️, ✅, 🛋️) to add personality and align with "{company_sentiment}".
-   Hashtags: Use 3-5 relevant hashtags. Mix popular ones from company's domain.
-   Call to Action: "Link in bio to explore our Free Features!", "Tag someone who needs this reminder!", "Save for later!".
-   Media (if Type B/C):
    -   Type B (Evocative): Aesthetically pleasing, high-quality images reflecting post sentiment.
    -   Type C (Informative): Visually engaging but calm background + captions with key messages from "{subject}". Text overlays should be stylish and readable.
    -   Media Prompt: If generating `platform_media_generation_prompt` for an {media_type_for_prompt}, 
    Craft a prompt to be fed into a Visual Generation Model. Aspect ratio for instagram Square 1:1. Emphasize "{company_sentiment}".

Your Tasks:
1.  Craft `platform_specific_text` (the caption) adhering to Instagram guidelines.
2.  If `post_type_decision` is 'B' or 'C', generate a detailed `platform_media_generation_prompt` for an {media_type_for_prompt}.

Output Format:
Return a single JSON object with the keys "platform_specific_text" and "platform_media_generation_prompt" (null if Type A).
"""

TWITTER_SYSTEM_PROMPT = """
You are a witty and concise Social Media Content Creator for {company_name}, specializing in Twitter (X).
Your goal is to adapt a core message into brief, impactful Tweets.

Company Name: {company_name}
Company Mission: {company_mission}
Company Sentiment: {company_sentiment}
Original Subject: {subject}
Post Type Decided by Strategist: {post_type_decision} (A: Text-only, B: Text + Evocative Media, C: Text + Informative Media)
Core Post Text Suggestion from Strategist: {core_post_text_suggestion}

Twitter (X) Specific Guidelines:
-   Tone: Conversational, direct, and encouraging. "{company_sentiment}" adapted for brevity.
-   Length: Max 150 characters. If {core_post_text_suggestion} is long, create a concise summary or key takeaway. NO THREADS for this task, aim for a single impactful tweet.
-   Hooks: Start with a strong hook related to "{subject}" e.g., "Get ready for {company_name}!", "Got some news for you", "Most of our services are Free"
-   Hashtags: Use 1-3 highly relevant hashtags (e.g., #{company_name_no_spaces}, other hashtags in sentiment))
-   Emojis: Use sparingly if they add clarity or save space (e.g., ✅, ➡️ ).
-   Call to Action: Clear and direct (e.g., "Stay tuned!", "Learn more [link if applicable]", "Whats your take on this?").
-   Media (if Type B/C):
    -   Type B (Evocative): Simple, clean images or short GIFs that are quickly digestible and convey warmth or readiness.
    -   Type C (Informative): Visually engaging but calm background + captions with key messages from "{subject}". Text overlays should be stylish and readable.
    -   Media Prompt: If generating `platform_media_generation_prompt` for an {media_type_for_prompt}, describe visuals for a fast-paced feed. Aspect ratios square 1:1.

Your Tasks:
1.  Craft `platform_specific_text` (Tweet content) adhering to Twitter guidelines.
2.  If `post_type_decision` is 'B' or 'C', generate a detailed `platform_media_generation_prompt` for an {media_type_for_prompt}.

Output Format:
Return a single JSON object with the keys "platform_specific_text" and "platform_media_generation_prompt" (null if Type A).
"""

FACEBOOK_SYSTEM_PROMPT = """
You are a versatile Social Media Content Creator for {company_name}, specializing in Facebook.
Your goal is to adapt a core message into an engaging Facebook post that encourages community interaction.

Company Name: {company_name}
Company Mission: {company_mission}
Company Sentiment: {company_sentiment}
Original Subject: {subject}
Post Type Decided by Strategist: {post_type_decision} (A: Text-only, B: Text + Evocative Media, C: Text + Informative Media)
Core Post Text Suggestion from Strategist: {core_post_text_suggestion}

Facebook Specific Guidelines:
-   Tone: Friendly, approachable, informative, and community-oriented. "{company_sentiment}" should be evident.
-   Length: Flexible. Can be a short update or a more descriptive post (2-4 paragraphs). First few lines are key.
-   Storytelling: Connect with the audience by highlighting the ease and benefit of {company_name}'s mission.
-   Questions: Engage users, e.g., "Are you excited", "Whats your experience with", "Would this help you?"
-   Emojis: Use emojis to add personality and align with "{company_sentiment}" (e.g., 😊, ✅).
-   Hashtags: Use 1-3 relevant hashtags (e.g., #{company_name_no_spaces}, other hashtags in sentiment)).
-   Call to Action: "Link in bio to explore our Free Features!", "Tag someone who needs this reminder!", "Save for later!".
-   Links: If provided with some clear link, use it accordingly.
-   Media (if Type B/C):
    -   Type B (Evocative): Aesthetically pleasing, high-quality images reflecting post sentiment.
    -   Type C (Informative): Visually engaging but calm background + captions with key messages from "{subject}". Text overlays should be stylish and readable.
    -   Media Prompt: If generating `platform_media_generation_prompt` for an {media_type_for_prompt}, 
    Craft a prompt to be fed into a Visual Generation Model. Aspect ratio for Facebook 16:9. Emphasize "{company_sentiment}".
    
Your Tasks:
1.  Craft `platform_specific_text` adhering to Facebook guidelines.
2.  If `post_type_decision` is 'B' or 'C', generate a detailed `platform_media_generation_prompt` for an {media_type_for_prompt}.

Output Format:
Return a single JSON object with the keys "platform_specific_text" and "platform_media_generation_prompt" (null if Type A).
"""

PLATFORM_PROMPT_MAP = {
    "linkedin": LINKEDIN_SYSTEM_PROMPT,
    "instagram": INSTAGRAM_SYSTEM_PROMPT,
    "twitter": TWITTER_SYSTEM_PROMPT,
    "facebook": FACEBOOK_SYSTEM_PROMPT,
}


async def run_platform_adaptation_agent(inputs: PlatformAgentInput) -> PlatformAgentOutput:
    target_platform_lower = inputs['target_platform'].lower()
    print(f"\n--- Running Layer 3: Platform Adaptation for {target_platform_lower} ---")

    system_prompt_template = PLATFORM_PROMPT_MAP.get(target_platform_lower)
    if not system_prompt_template:
        raise ValueError(f"No system prompt defined for platform: {inputs['target_platform']}")

    media_type_for_prompt = "image"
    if inputs['post_type_decision'] == 'D':
        media_type_for_prompt = "video"

    # Prepare company_name_no_spaces for hashtag usage
    company_name_no_spaces = inputs["company_name"].replace(" ", "")

    formatted_system_prompt = system_prompt_template.format(
        company_name=inputs["company_name"],
        company_mission=inputs["company_mission"],
        company_sentiment=inputs["company_sentiment"],
        subject=inputs["subject"],
        post_type_decision=inputs["post_type_decision"],
        core_post_text_suggestion=inputs["core_post_text_suggestion"],
        target_platform=inputs["target_platform"],  # The prompt itself might use this for clarity
        media_type_for_prompt=media_type_for_prompt,
        company_name_no_spaces=company_name_no_spaces  # For easy hashtag creation
    )

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=formatted_system_prompt),
        HumanMessage(
            content=f"Please generate the tailored content for {inputs['target_platform']}. Remember to output in the specified JSON format.")
    ])

    chain = prompt_template | platform_llm | JsonOutputParser()

    try:
        response: PlatformAgentOutput = await chain.ainvoke({})
        print(
            f"Platform Agent ({inputs['target_platform']}) Text: {response.get('platform_specific_text', '')[:100]}...")
        if response.get('platform_media_generation_prompt'):
            print(
                f"Platform Agent ({inputs['target_platform']}) Media Prompt: {response['platform_media_generation_prompt'][:100]}...")
        return response
    except Exception as e:
        print(f"Error in Platform Adaptation for {inputs['target_platform']}: {e}")
        # You might want to return a default error structure or re-raise
        # For now, re-raising to halt on error.
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

    platform_agent_outputs_with_name = await asyncio.gather(*(task for _, task in platform_adaptation_tasks))

    platform_outputs_map: Dict[str, PlatformAgentOutput] = {}
    for i, output in enumerate(platform_agent_outputs_with_name):
        platform_name = platform_adaptation_tasks[i][0]
        platform_outputs_map[platform_name] = output

    media_generation_tasks = []
    platform_names_needing_media = []  # Keep track of which platforms had a media prompt

    if decided_post_type in ["B", "C"]:
        for platform_name in target_platforms:
            platform_output = platform_outputs_map.get(platform_name)
            if not platform_output:
                print(f"Warning: No platform output found for {platform_name}. Skipping media generation.")
                continue

            media_prompt = platform_output.get("platform_media_generation_prompt")
            if media_prompt:
                media_type_to_generate: Literal["image", "video"] = "image"
                if decided_post_type == "C":
                    media_type_to_generate = "video"

                media_generation_tasks.append(
                    generate_media_content(media_prompt, media_type_to_generate)
                )
                platform_names_needing_media.append(platform_name)  # Add platform_name to this list
            # No warning here if media_prompt is None, as Type A posts won't have it.
            # The platform agent itself is responsible for returning null for Type A.

    generated_media_assets_list: List[Optional[MediaAsset]] = []
    if media_generation_tasks:
        generated_media_assets_list = await asyncio.gather(*media_generation_tasks)

    # Map generated media back to the correct platform
    generated_media_map: Dict[str, MediaAsset] = {}
    for i, platform_name_with_media in enumerate(platform_names_needing_media):
        if i < len(generated_media_assets_list):
            generated_media_map[platform_name_with_media] = generated_media_assets_list[i]

    final_posts: List[FinalGeneratedPost] = []
    for platform_name in target_platforms:
        platform_output = platform_outputs_map.get(platform_name)
        if not platform_output:
            print(f"Warning: No platform output found for {platform_name} during final assembly. Skipping.")
            continue

        current_media_asset: Optional[MediaAsset] = None
        if decided_post_type in ["B", "C"] and platform_output.get("platform_media_generation_prompt"):
            current_media_asset = generated_media_map.get(platform_name)  # Get media using platform name

        final_posts.append(FinalGeneratedPost(
            platform=platform_name,
            post_type=decided_post_type,
            text=platform_output["platform_specific_text"],
            media_asset=current_media_asset
        ))

    print("\n✅ Social Media Post Generation Pipeline Complete! ✅")
    return final_posts


# --- Example Usage ---
async def main_orchestration():
    sample_posts_history: List[PostHistoryEntry] = [
        {"post_type": "A", "count": 7, "score": 8},
        {"post_type": "A", "count": 7, "score": 8},
        {"post_type": "A", "count": 7, "score": 8}

    ]

    target_platforms = ["linkedin", "instagram", "twitter", "facebook"]

    generated_posts_list = await generate_social_media_posts(
        subject=POST_SUBJECT,
        target_platforms=target_platforms,
        requirements=None,
        posts_history=sample_posts_history
    )

    print("\n--- Individual Generated Posts (Summary) ---")
    for i, post in enumerate(generated_posts_list):
        print(f"\n--- Post {i + 1} for {post['platform']} (Type: {post['post_type']}) ---")
        print(f"Text: {post['text'][:250]}...")  # Print summary
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
        final_generated_posts = asyncio.run(main_orchestration())
        final_json_output = json.dumps(final_generated_posts, indent=4)
        print("\n\n--- Final JSON Output ---")
        print(final_json_output)
        output_filename = "social_media_posts_output.json"
        with open(output_filename, "w", encoding='utf-8') as f:  # Added encoding
            f.write(final_json_output)
        print(f"\n✅ Final JSON output saved to {output_filename}")