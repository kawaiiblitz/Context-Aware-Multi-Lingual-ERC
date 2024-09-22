import openai

class Client:
    def __init__(self, api_key=None):
        self.api_key = api_key

    def call_model(self, body, model_id):
        pass

class LlamaCpp(Client):
    def __init__(self, base_url=""):
        super().__init__(api_key)
        self.base_url = base_url
        self.client = openai.OpenAI(base_url=base_url, api_key=self.api_key)

    def call_model(self, text, model_id="gpt-3.5-turbo"):

        examples = [
            "Classes: [happy, excited, sad, neutral, frustrated, angry]",
            "Text: Si. Eso es increible.",
            "Class: excited",
            "Text: Dios mio. Que vas a hacer? [RISAS]",
            "Class: excited",
            "Text: No se. Yo tambien lo solicite. Ellos no...",
            "Class: sad",
            "Text: [RESPIRANDO]",
            "Class: frustrated",
            "Text: No. Numero equivocado. [RISAS]",
            "Class: happy",
            "Text: Y yo que?",
            "Class: angry",
            "Text: No seas estupido.",
            "Class: angry"
        ]
    
        examples_text = "\n".join(examples)
        content = f"""{examples_text}

Text: {text}
Classify the text into one of the above classes. Provide only the class name as the output."""

        completion = self.client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."},
                {"role": "user", "content": content}
            ]
        )

        response_body = completion.choices[0].message.content.strip()

        valid_labels = {"happy", "excited", "sad", "neutral", "frustrated", "angry"}
        if response_body not in valid_labels:
            response_body = "neutral" 

        return response_body
