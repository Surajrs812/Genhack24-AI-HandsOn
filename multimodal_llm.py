from pathlib import Path
import base64
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()
import os

llm = ChatAnthropic(temperature=0, api_key=os.environ.get("ANTHROPIC_API_KEY"), model_name="claude-3-sonnet-20240229")
print(llm.invoke("Give me information of 4 vedas.").content)

print("\n\nNow. We will look at multimodality.\n")
img_path = Path("data\Animal.jpg")
img_base64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
messages = [
    HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}",  
                },
            },
            {"type": "text", "text": "Can you tell me about this animal."},
        ]
    )
]

print(llm.invoke(messages).content)
