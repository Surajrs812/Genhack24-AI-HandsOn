from langchain_groq import ChatGroq
from langchain_experimental.tools import Tool
from langchain_experimental.agents import create_openai_functions_agent
import os
import requests
import math
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the ChatGroq language model
llm = ChatGroq(api_key=os.environ.get("GROQ_API_KEY"), temperature=0.2)

# Tool 1: Perform advanced mathematical operations
def calculate_complex_equation(equation: str):
    """
    Evaluate complex mathematical equations using Python's eval function.
    WARNING: Be cautious when using eval to avoid executing arbitrary code.
    """
    try:
        result = eval(equation, {"__builtins__": None}, {"math": math})
        return f"The result of the equation '{equation}' is {result}."
    except Exception as e:
        return f"Error calculating equation: {e}"

math_tool = Tool(
    name="ComplexMathTool",
    description="This tool performs advanced mathematical computations. Pass a valid equation as input.",
    func=calculate_complex_equation
)

# Tool 2: Get real-time data from a public API (like weather data)
def fetch_weather(city: str):
    """
    Fetch weather data from a public API (e.g., OpenWeather).
    Note: Replace 'YOUR_API_KEY' with a valid key.
    """
    API_KEY = os.getenv('WEATHER_API_KEY')  # Load from .env
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    
    try:
        response = requests.get(url)
        data = response.json()
        if data["cod"] != 200:
            return f"Error: {data['message']}"
        
        # Extract important weather information
        weather_desc = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        
        return f"Weather in {city}: {weather_desc}, Temperature: {temperature}°C, Humidity: {humidity}%"
    
    except Exception as e:
        return f"Error fetching weather data: {e}"

weather_tool = Tool(
    name="WeatherTool",
    description="Fetches current weather information for a given city.",
    func=fetch_weather
)

# Tool 3: Summarize text from a file
def summarize_text_file(file_path: str):
    """
    Summarizes the content of a text file.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            # For simplicity, let's just summarize by truncating (you can add more advanced summarization)
            summary = content[:300] + '...' if len(content) > 300 else content
            return f"Summary of the file:\n{summary}"
    except Exception as e:
        return f"Error reading file: {e}"

file_summary_tool = Tool(
    name="FileSummaryTool",
    description="Reads and summarizes a text file.",
    func=summarize_text_file
)

# Create the complex agent with multiple tools
def create_complex_chatgroq_agent():
    """
    Creates a ChatGroq-based agent capable of performing advanced actions:
    - Complex mathematical computations
    - Fetching real-time weather data
    - Summarizing text files
    """
    agent_executor = create_openai_functions_agent(
        llm,  # Use the ChatGroq model
        tools=[math_tool, weather_tool, file_summary_tool],  # Add multiple tools to the agent
        verbose=True  # Enable detailed logging of the agent's actions
    )
    return agent_executor

# Query the agent and extract its response
def query_agent(agent, query):
    """
    Query the agent with a natural language input and return the response.
    """
    response = agent.invoke(query)
    return response

# Create the agent
agent = create_complex_chatgroq_agent()

# Example queries for complex actions

# 1. Ask the agent to solve a complex mathematical equation
math_query = "Solve the equation: 3 * math.sin(math.radians(30)) + math.log(10)"
print(query_agent(agent, math_query))

# 2. Ask the agent to fetch real-time weather data
weather_query = "Get the weather for New York."
print(query_agent(agent, weather_query))

# 3. Ask the agent to summarize a text file (assuming the file exists)
file_query = "Summarize the contents of 'sample.txt'."
print(query_agent(agent, file_query))
