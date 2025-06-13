from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from dotenv import dotenv_values

# Load .env file as a dictionary (without affecting os.environ)
config = dotenv_values(".env")

# Access the key directly from the config dictionary
GEMINI_API_KEY_VALUE = config.get("GEMINI_API_KEY")

if not GEMINI_API_KEY_VALUE:
    raise ValueError("GEMINI_API_KEY is not set or not found in .env file.")

print(GEMINI_API_KEY_VALUE)

client = genai.Client(api_key=GEMINI_API_KEY_VALUE)


response = client.models.generate_images(
    model='imagen-3.0-generate-002',
    prompt='Robot holding a red skateboard',
    config=types.GenerateImagesConfig(
        number_of_images= 1
    )
)
for generated_image in response.generated_images:
  image = Image.open(BytesIO(generated_image.image.image_bytes))
  image.show()