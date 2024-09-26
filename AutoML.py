from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the ChatGroq language model with the API key and custom temperature
llm = ChatGroq(api_key=os.environ.get("GROQ_API_KEY"))

# Read the dataset into a pandas DataFrame
df = pd.read_csv("data\Bengaluru_House_Data.csv")

# Function to create a pandas dataframe agent
def create_pandas_agent(llm, df):
    """
    Creates a Pandas DataFrame agent using the provided language model and DataFrame.
    The agent is capable of performing ML and data operations based on queries.
    """
    agent_executor = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="tool-calling",  # Specifies that the agent can call tools or perform computations
        verbose=True,  # Enable detailed logs of the agent's activity
        allow_dangerous_code=True,  # Allow running complex code
        include_df_in_prompt=True  # Include a description of the DataFrame in the agent's context
    )
    return agent_executor

# Function to query the agent and extract output
def query_data(agent, query):
    """
    Queries the agent with a specific question and extracts the output from the agent's response.
    """
    response = agent.invoke(query)
    # Extract the 'output' from the response, or return 'No output found' if absent
    output_value = response.get('output', 'No output found')
    return output_value


# Create the agent by calling the function
agent = create_pandas_agent(llm, df)

# Query the agent for a house price prediction
print(query_data(agent, "You are an AutoML Expert, use any ML operations Predict a price for a 3BHK House in Electronic city."))
