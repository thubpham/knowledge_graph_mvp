from google import genai

class LLMClient:
    def __init__(self):
        self.client = genai.Client()

    def generate_gemini(self, prompt: str):
        response = self.client.models.generate_content(
            model = "gemini-2.5-flash",
            contents = prompt
        )
        return response.text