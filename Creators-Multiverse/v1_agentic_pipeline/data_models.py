# data_models.py
from typing import Dict, List, TypedDict, Optional, Literal, Union


class PostHistoryEntry(TypedDict):
    post_type: Literal["A", "B", "C"]  # Future: "D", "E" for video
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
    post_type: Literal["A", "B", "C"]  # Future: "D", "E"
    core_post_text: str


class PlatformAgentInput(TypedDict):
    company_name: str
    company_mission: str
    company_sentiment: str
    subject: str
    post_type_decision: Literal["A", "B", "C"]  # Future: "D", "E"
    core_post_text_suggestion: str
    target_platform: str


class PlatformAgentOutput(TypedDict):
    platform_specific_text: str
    platform_media_generation_prompt: Optional[str]  # Prompt for image/video


class SavedMediaAsset(TypedDict):
    type: Literal["image", "video"]  # Currently only 'image'
    file_path: str  # Path to the locally saved file


class FinalGeneratedPost(TypedDict):
    platform: str
    post_type: Literal["A", "B", "C"]  # Original decision from Layer 2
    text_file_path: str  # Path to the .txt file
    media_asset: Optional[SavedMediaAsset]  # Details of the saved media
    original_text_content: str  # The actual text content
    media_generation_prompt_used: Optional[str]  # The prompt used for media, if any
