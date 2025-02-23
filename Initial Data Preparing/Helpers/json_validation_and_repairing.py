import re
import json

data = """
[

{"category": "Food & Digestive System", "chunk": "Long community time with Mister,\\nIgnited fire about OutlierAI, maybe some beautiful things will come.\\nAs evening comes as he leaves, we're so hungry once again.\\nEat our food bowl, pretty big excess -\\nThen the 2h later wake-up to eating croissant and milk.\\nWhy did we eat tho, my dear night brother?\\nIs it extraordinarily complex to answer that.\\nBut dhamma test 3&4 clear peak negative score for croissant  + milk.\\nWell, didn't you know that, old friend? From your brothers\\nBut what made us buy it, eat as we eate, sleep as we slept.. well.. that is too complex.\\nAlso dukkha cultivated by tomorrow's lack of gas (despite trust), work stuff, pelicanu little friendly mocking with Mister..\\nAnother veeery tired night in which we cannot really rise when body asks to, dream of the Phisolopher Pelicanu.\\nBut still 10x better than before.\\nNot too much, maybe half with milk?\\nIts so complex man... it's so complex......\\nAs my dear wise grandmother always quotes from the great Romanian poet and philosopher Eminescu:\\n\\"Don't seek to understand these Laws, for you will go Mad and still don't get them.\\"\\nThe low-level..\\nWake up at 5.\\nThere is no Recovery time desite heavy'ish sleep, possibly because last meal has not been fueled by Night's fear and food."}
,

{"category": "Into the Soul", "chunk": "The Alchemist talking to the wind and desert. \"Well, what good would it be to you if you had to die?\" the alchemist answered. \"Your money saved us for three days. It's not often that money saves a person's life.\" But the boy was too frightened to listen to words of wisdom. He had no idea how he was going to transform himself into the wind. He wasn’t an alchemist! The alchemist asked one of the soldiers for some tea, and poured some on the boy's wrists. A wave of relief washed over him, and the alchemist muttered some words that the boy didn’t understand. \"Don't give in to your fears,\" said the alchemist, in a strangely gentle voice. \"If you do, you won’t be able to talk to your heart.\" \"But I have no idea how to turn myself into the wind.\" \"If a person is living out his Personal Legend, he knows everything he needs to know. There is only one thing that makes a dream impossible to achieve: the fear of failure.\" \"I'm not afraid of failing. It's just that I don’t know how to turn myself into the wind.\" \"Well, you’ll have to learn; your life depends on it.\" \"But what if I can’t?\" \"Then you’ll die in the midst of trying to realize your Personal Legend. That’s a lot better than dying like millions of other people, who never even\"}

]
"""

# Replace isolated backslashes using regex
data = re.sub(r'(?<!\\)\\(?![nrt"])', r'\\\\', data)


# Handle "purpose/environment" or similar patterns
def replace_word_slash_word(text):
    """
    Replace patterns like 'purpose/environment' with 'purpose and environment'.
    """

    def replacement(match):
        parts = match.group(0).split("/")
        return f"{parts[0]} and {parts[1]}"

    pattern = r'\b\w+/\w+\b'
    return re.sub(pattern, replacement, text)


if __name__ == "__main__":

    # Apply the replacement function
    data = replace_word_slash_word(data)

    print(data)

    # Parse JSON
    json_data = json.loads(data)
    print(json_data)
