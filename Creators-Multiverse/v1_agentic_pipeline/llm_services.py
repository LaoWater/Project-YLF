# llm_services.py
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from config import (
    OPENAI_API_KEY_VALUE,
    DECISION_LLM_MODEL,
    PLATFORM_LLM_MODEL
)
from data_models import (
    Layer2Input, Layer2Output,
    PlatformAgentInput, PlatformAgentOutput
)

# --- LLM Initialization ---
decision_llm = ChatOpenAI(
    api_key=OPENAI_API_KEY_VALUE,
    model=DECISION_LLM_MODEL,
    temperature=1,
)

platform_llm = ChatOpenAI(
    api_key=OPENAI_API_KEY_VALUE,
    model=PLATFORM_LLM_MODEL,
    temperature=1.1,
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
    chain = prompt_template | decision_llm | JsonOutputParser()
    raw_llm_output_str = ""
    try:
        # To see the raw output before parsing (for debugging)
        llm_response_message = await (prompt_template | decision_llm).ainvoke({})
        raw_llm_output_str = llm_response_message.content
        print(f"\n--- Layer 2: Raw LLM Response ---\n{raw_llm_output_str}\n--- End of Raw LLM Response ---")

        response: Layer2Output = JsonOutputParser().parse(raw_llm_output_str) # type: ignore
        print(f"Layer 2 Decision (Parsed): Post Type: {response.get('post_type')}, Core Text: {response.get('core_post_text', '')[:100]}...")
        return response
    except Exception as e:
        print(f"Error in Layer 2: {e}")
        if raw_llm_output_str:
            print(f"Problematic raw LLM output on error:\n{raw_llm_output_str}")
        raise

# --- Layer 3: Platform Adaptation LLM Prompts ---
LINKEDIN_SYSTEM_PROMPT = """
You are an expert Social Media Content Creator for {company_name}, specializing in LinkedIn.
Your goal is to adapt a core message into a professional and engaging LinkedIn post.

Company Name: {company_name}
Company Mission: {company_mission}
Company Sentiment: {company_sentiment}
Original Subject: {subject}
Post Type Decided by Strategist: {post_type_decision} (A: Text-only, B: Text + Evocative Photo, C: Text + Informative Photo)
Core Post Text Suggestion from Strategist: {core_post_text_suggestion}

LinkedIn Specific Guidelines:
-   Tone: Professional, insightful, authoritative, value-driven. Align with "{company_sentiment}".
-   Hashtags: Use 3-5 relevant, professional hashtags. Consider #{company_name_no_spaces}.
-   Media (if Type B/C):
    -   Media Prompt: If `post_type_decision` is 'B' or 'C' and you deem media appropriate for LinkedIn, craft a `platform_media_generation_prompt` for an {media_type_for_prompt}. Aspect ratio 1.91:1 or 1:1 (square for carousel). For a single image, prefer landscape or square that works well in feed.

Your Tasks:
1.  Craft `platform_specific_text` for LinkedIn.
2.  If `post_type_decision` is 'B' or 'C', AND you decide media is suitable for this specific LinkedIn post, generate `platform_media_generation_prompt`. Otherwise, set it to null.

Output Format:
Return a single JSON object with keys "platform_specific_text" (string) and "platform_media_generation_prompt" (string or null).
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
-   Tone: Engaging, friendly, authentic, visually descriptive. "{company_sentiment}" should shine.
-   Hashtags: Use 5-10 relevant hashtags. Mix popular and niche. Include #{company_name_no_spaces}.
-   Media (if Type B/C): Instagram is highly visual.
    -   Media Prompt: If `post_type_decision` is 'B' or 'C', craft `platform_media_generation_prompt` for an {media_type_for_prompt}. Aspect ratio square 1:1 or portrait 4:5. Emphasize "{company_sentiment}".

Your Tasks:
1.  Craft `platform_specific_text` (caption) for Instagram.
2.  If `post_type_decision` is 'B' or 'C', generate `platform_media_generation_prompt`. If Type A, this must be null.

Output Format:
Return a single JSON object with keys "platform_specific_text" (string) and "platform_media_generation_prompt" (string or null).
"""

TWITTER_SYSTEM_PROMPT = """
You are a charming and concise Social Media Content Creator for {company_name}, specializing in Twitter (X).
Your goal is to adapt a core message into brief, impactful Tweets.

Company Name: {company_name}
Company Mission: {company_mission}
Company Sentiment: {company_sentiment}
Original Subject: {subject}
Post Type Decided by Strategist: {post_type_decision} (A: Text-only, B: Text + Evocative Media, C: Text + Informative Media)
Core Post Text Suggestion from Strategist: {core_post_text_suggestion}

Twitter (X) Specific Guidelines:
-   Tone: Conversational, direct, encouraging. "{company_sentiment}" adapted for brevity.
-   Length: Max 280 characters.
-   Hashtags: Use 1-3 highly relevant hashtags (e.g., #{company_name_no_spaces}).
-   Media (if Type B/C):
    -   Media Prompt: If `post_type_decision` is 'B' or 'C', craft `platform_media_generation_prompt` for an {media_type_for_prompt}. Aspect ratio 16:9 or 1:1.

Your Tasks:
1.  Craft `platform_specific_text` (Tweet content).
2.  If `post_type_decision` is 'B' or 'C', generate `platform_media_generation_prompt`. If Type A, this must be null.

Output Format:
Return a single JSON object with keys "platform_specific_text" (string) and "platform_media_generation_prompt" (string or null).
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
-   Tone: Friendly, approachable, informative, community-oriented. "{company_sentiment}" should be evident.
-   Hashtags: Use 1-3 relevant hashtags (e.g., #{company_name_no_spaces}).
-   Media (if Type B/C):
    -   Media Prompt: If `post_type_decision` is 'B' or 'C', craft `platform_media_generation_prompt` for an {media_type_for_prompt}. Aspect ratio 1.91:1 (landscape) or 1:1 (square). Emphasize "{company_sentiment}".

Your Tasks:
1.  Craft `platform_specific_text` for Facebook.
2.  If `post_type_decision` is 'B' or 'C', generate `platform_media_generation_prompt`. If Type A, this must be null.

Output Format:
Return a single JSON object with keys "platform_specific_text" (string) and "platform_media_generation_prompt" (string or null).
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

    system_prompt_template_str = PLATFORM_PROMPT_MAP.get(target_platform_lower)
    if not system_prompt_template_str:
        raise ValueError(f"No system prompt defined for platform: {inputs['target_platform']}")

    media_type_for_prompt: Literal["image", "video"] = "image" # Default to image for B/C
    # Future: if inputs['post_type_decision'] in ['D', 'E']: media_type_for_prompt = "video"

    company_name_no_spaces = inputs["company_name"].replace(" ", "")

    formatted_system_prompt = system_prompt_template_str.format(
        company_name=inputs["company_name"],
        company_mission=inputs["company_mission"],
        company_sentiment=inputs["company_sentiment"],
        subject=inputs["subject"],
        post_type_decision=inputs["post_type_decision"],
        core_post_text_suggestion=inputs["core_post_text_suggestion"],
        target_platform=inputs["target_platform"],
        media_type_for_prompt=media_type_for_prompt,
        company_name_no_spaces=company_name_no_spaces
    )

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=formatted_system_prompt),
        HumanMessage(content=f"Please generate the tailored content for {inputs['target_platform']}. Remember to output in the specified JSON format.")
    ])
    chain = prompt_template | platform_llm | JsonOutputParser()
    raw_llm_output_str = ""
    try:
        # To see the raw output before parsing (for debugging)
        llm_response_message = await (prompt_template | platform_llm).ainvoke({})
        raw_llm_output_str = llm_response_message.content
        print(f"\n--- Platform Agent ({inputs['target_platform']}): Raw LLM Response ---\n{raw_llm_output_str}\n--- End Raw LLM Response ---")

        response: PlatformAgentOutput = JsonOutputParser().parse(raw_llm_output_str) # type: ignore

        print(f"Platform Agent ({inputs['target_platform']}) Text: {response.get('platform_specific_text', '')[:100]}...")
        if response.get('platform_media_generation_prompt'):
            print(f"Platform Agent ({inputs['target_platform']}) Media Prompt: {response['platform_media_generation_prompt'][:100]}...")
        return response
    except Exception as e:
        print(f"Error in Platform Adaptation for {inputs['target_platform']}: {e}")
        if raw_llm_output_str:
            print(f"Problematic raw LLM output on error for {inputs['target_platform']}:\n{raw_llm_output_str}")
        raise