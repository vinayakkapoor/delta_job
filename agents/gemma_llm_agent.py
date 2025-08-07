from google import genai
from google.genai import types
import os

SYSTEM_PROMPT = """
You are a highly specialized AI for extracting information from unstructured web content. You will be given the raw text scraped from a company's careers page.

Your task is to intelligently identify and extract all distinct job and internship titles. Be aware that titles appear in many different formats—they can be in headings, links, lists, or within paragraph sentences.

Instructions

1. Extract the Full Title: Capture the most complete and specific title available. For example, extract "Senior Software Engineer (Backend)" instead of just "Software Engineer".

2. Ignore Non-Title Text: Do not extract generic, repeated text like "Read More", "Apply Now", department names (unless part of the title), or locations listed separately.

3. Strict Output Format: Your entire response must be a single Python list of strings. Do not add any conversational text, explanations, or code formatting like markdown.

4. Handle No Results: If you cannot find any titles, you must return an empty list: [].
"""

class LLMAgent():
    def __init__(self, model='gemma-3-27b-it'):
        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        self.model_name_str = model

    @property
    def model_name(self):
        return self.model_name_str

    def run(self, text):
        prompt = f"{SYSTEM_PROMPT}\n\n user: {text}"
        
        contents = []        
        contents.append(
            types.Content(
                role='user',
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            )
        )

        try:
            response = self.client.models.generate_content(model=self.model_name_str, contents=contents)
        except:
            return None

        return response.text