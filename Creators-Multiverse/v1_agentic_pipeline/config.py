# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys & Environment ---
OPENAI_API_KEY_VALUE = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY_VALUE:
    raise ValueError("OPENAI_API_KEY environment variable is not set or not found in .env file.")

# --- Company & Request Configuration ---
COMPANY_NAME = "Creators Multiverse"
COMPANY_MISSION = "Empowering creators to build their digital presence with AI-powered tools that transform ideas into viral content across platforms"
COMPANY_SENTIMENT = ("Inspirational & Empowering. Cosmic/Magical Theme yet not too much."
                     "The brand positions itself as a creative partner that amplifies human creativity rather than replacing it.")
# This will be passed to the main orchestrator, but a default can be here
DEFAULT_POST_SUBJECT = "Hello World! Intro post about our company, starting out, vision, etc"

# --- LLM Model Configuration ---
DECISION_LLM_MODEL = "gpt-4o"
PLATFORM_LLM_MODEL = "gpt-4.1-2025-04-14"

# --- Output Configuration ---
BASE_OUTPUT_FOLDER = "generated_posts" # A root folder for all platform outputs
SUPPORTED_PLATFORMS = ["linkedin", "instagram", "twitter", "facebook"]

# --- Media Configuration ---
# For now, we are dealing with images. Video types (D, E) are for future.
IMAGE_FILE_EXTENSION = "jpg"