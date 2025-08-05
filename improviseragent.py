from openai import OpenAI, AsyncOpenAI
from agents import Agent, Tool, Runner, trace, RunConfig, ModelProvider, ModelSettings, OpenAIChatCompletionsModel

from pypdf import PdfReader
from improvgame import ImprovGame
import os

os.environ["OPENAI_API_KEY"] = "ollama"



class ImproviserAgent:

    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.chat_sessions = {}
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        self.asyncClient = AsyncOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

        self.model_settings = ModelSettings(
            model=self.model,
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

        class OllamaModelProvider(ModelProvider):
            def get_model(self, model_name: str):
                return OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=AsyncOpenAI(
                        base_url="http://localhost:11434/v1",
                        api_key="ollama"
                    )
                )

        self.run_config = RunConfig(
            model_settings=self.model_settings,
            model_provider=OllamaModelProvider()
        )

        reader = PdfReader("data/Living-Playbook.pdf")
        self.text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.text += text


    def client(self):
        return self.client

    def model(self):
        return self.model

    def __get_improv_game_decider_tool__(self, messages: list) -> Tool:
        """
        This tool is used to decide the game of improv to play.
        """
        system_prompt = f"""
        You are a friendly and helpful tool. You know a lot about improv. You have access 
        to the book "Living Playbook" and you can use it to pick a two person game for the user to play with another agent.
        You should also return a description of the game and the rules of the game and also the context of why you picked the game.
        """
        system_prompt += f"""
        Here is the context of the book:
        {self.text}
        """

        improv_game_decider_agent = Agent(
            name="ImprovGameDeciderAgent",
            instructions=system_prompt,
            model=self.model,
            model_settings=self.model_settings,
            output_type=ImprovGame,
        )
        return improv_game_decider_agent.as_tool(tool_name="improv_game_decider", tool_description="This tool is used to decide the game of improv to play.")

    def __get_improv_game_player_agent__(self, messages: list) -> Agent:
        """
        Plays a game of improv with the user.
        """
        SYSTEM_PROMPT = "You are Harold, an improv partner. You are playing a game of improv with the user." \
        "You are polite, respectful, and aim to provide concise responses of less than 20 words." \
        "You should ask the user if they want to play a game. If they do then you should use the improv_game_decider tool to pick a game." \
        "You should then play the game with the user. If the user wants to change the game then you should use the improv_game_decider tool to pick a new game." \

        return Agent(
            name="GameAgent",
            instructions=SYSTEM_PROMPT,
            tools=[self.__get_improv_game_decider_tool__(messages)],
            model=self.model,
            model_settings=self.model_settings,
        )


    def __get_session_history_messages__(self, session_id: str) -> list:
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
        messages = self.__get_session_history_messages__(session_id)

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

    async def play_game(self, text: str) -> str:

        client1 = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        response1 = client1.chat.completions.create(
                         model="llama3.2",
                         messages=[{"role": "user", "content": "What’s the capital of France?"}]
                    )

        print(response1.choices[0].message.content)

        print(f"[yellow]Playing a game of improv with the user...[/yellow]")

        session_id = "improviser_agent_session"

       # Get current session history
        messages = self.__get_session_history_messages__(session_id)

        # Add the current human input to the history
        messages.append({"role": "user", "content": text})

        game_agent = self.__get_improv_game_player_agent__(messages)

        with trace("Play"):
            result = await Runner.run(game_agent, messages, run_config=self.run_config)
            response = result.final_output

        #Add the AI's response to the history
        messages.append({"role": "assistant", "content": response})
        print(f"[green]Response: {response}[/green]")

        return response