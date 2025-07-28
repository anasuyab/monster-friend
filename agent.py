from openai import OpenAI

class ImproviserAgent:

    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.chat_sessions = {}
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
    def client(self):
        return self.client

    def model(self):
        return self.model

    def get_session_history_messages(self, session_id: str) -> list:
        """Get or create chat history for a session as a list of message dicts."""
        SYSTEM_PROMPT = "You are an improviser" \
        "You are polite, respectful, and aim to provide concise responses of less than 20 words." \
        "You are an improv partner. You are playing a game of improv with the user." \
    
        if session_id not in self.chat_sessions:
            # Initialize with the system message
            self.chat_sessions[session_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
        return self.chat_sessions[session_id]

    def generate_response(self, text: str) -> str:
        """
        Generates a response to the given text using Ollama via the OpenAI SDK.
        Manually manages chat history.
        """
        session_id = "improviser_session"

       # Get current session history
        messages = self.get_session_history_messages(session_id)

        # Add the current human input to the history
        messages.append({"role": "user", "content": text})

        try:
            # Use the OpenAI client's chat completions
            response = self.client.chat.completions.create(
                model=self.model, # Use the model argument for Ollama model (e.g., "gemma3")
                messages=messages,
                temperature=0.7, # You can add other parameters like temperature, max_tokens etc.
                stream=False # Set to True if you want to stream the response
            )

            llm_response_content = response.choices[0].message.content.strip()

            # Add the AI's response to the history
            messages.append({"role": "assistant", "content": llm_response_content})

            return llm_response_content

        except Exception as e:
            print(f"[red]Error getting LLM response from Ollama (via OpenAI SDK): {e}")
            # Optionally, remove the last user message if the LLM call failed to maintain consistent history
            if messages and messages[-1]["role"] == "user":
                messages.pop()
            return "I'm sorry, I couldn't generate a response at this moment."