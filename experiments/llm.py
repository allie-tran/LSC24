# auto-reload

# %load_ext autoreload
# %autoreload 2

import asyncio
import os
from typing import List

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import (ChatCompletionMessageParam,
                               ChatCompletionSystemMessageParam,
                               ChatCompletionUserMessageParam)

#|%%--%%| <ncdWWwxOEY|cE58yjx8NV>

load_dotenv()

# Set up ChatGPT generation model
OPENAI_API=os.environ.get("OPENAI_API", "")
MODEL_NAME=os.environ.get("MODEL_NAME", "")

class LLM():
    def __init__(self):
        self.client = AsyncOpenAI(api_key=OPENAI_API)

    async def generate(self, messages: List[ChatCompletionMessageParam]):
        request = await self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True) # type: ignore
        async for chunk in request:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def get_template_messages(self) -> List[ChatCompletionMessageParam]:
        message = ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant to help me answer questions in a lifelog dataset. I will give you information and the system constraints and you will suggest me how to do it.")
        return [message]

#|%%--%%| <cE58yjx8NV|lJGUaxXoc6>

# Test message
model = LLM()

async def generate(messages: List[ChatCompletionMessageParam]):
    async for response in model.generate(messages):
        print(response, end='')
    return

def get_recommendation(question: str):
    prompt = """I need to find the answer for this question using my lifelog retrieval system. In my system, a flow of processes is needed:
1. Segmentation: this function takes two parameters: max_time, time_gap, and loc_change, where max_time is the maximum time for each segment, time_gap is the maximum time gap between two segments, and loc_change is the type of location change (semantic_location, city, country, continent). The function returns a list of segments, where each segment is a list of events.
2. Retrieval: this function takes a list of segments and a question. It returns a list of events that are relevant to the question. The function takes the top-K events that are relevant to the question.
3. Extraction: this function takes a list of events and a question. It returns the answer to the question. The function extracts the information from the events to answer the question.
4. Answering: this function takes the answer and the question. It returns the answer to the question.
5. Post-processing: re-organize the events (merge, split, or filter) and the answer to the question. Events with the same answers can be grouped together (if it makes sense).

For example, if the question is "What is my favourite airlines to fly with in 2019?", this is what I'm looking for:
- Segmentation: max_time=1 day, time_gap=1 day, loc_change=city
- Retrieval: query="airlines name on boarding pass or brochure', K=50
- Extraction: metadata=["start_city", "end_city"]
- Answering: needs Visual Question Answering=yes, needs OCR=yes, expected answer type=a name, possible answers=["Delta", "United", "American", "Southwest", "JetBlue"], sort by time=no
- Post-processing: sort=time, group=airlines

Now, the question is "{question}". I need you to define these paramters:
Please provide the following JSON structure:
```
{{
    "segmentation": {{
        "max_time": [a time unit in the following: "year", "month", "week", "day", "hour"],
        "time_gap": [in hours],
        "loc_change": [a location unit,  one of the following: "country", "city", "location_name", "continent"]
    }},
    "retrieval": {{
        "search query": [a search query to find the events],
        "K": [number of events to retrieve and extract answers from]
    }},
    "extraction": {{
        "metadata": [a list of metadata to extract from each event, one of the following: "start_time", "end_time", "semantic_location", "duration", "country", "city", "continent" that might be useful to answer the question],
        "needs Visual Question Answering": [true/false],
        "needs OCR": [true/false],
    }},
    "answering": {{
        "expected answer type": "explanation of what the answer should look like",
        "possible answers": [a list of possible answers]
        }},
    "post-processing": {{
        "group": [a way to group the events, one of the following: any time unit, any location unit, "answer"],
        "sort": [a way to sort the events, one of the following: any time unit, any location_unit, "most_common_answer"]
        }}
}}
No explanation is needed. Just provide the JSON structure.
```
    """
    prompt = prompt.format(question=question)
    messages = model.get_template_messages()
    messages.append(ChatCompletionUserMessageParam(role="user", content=prompt))
    asyncio.run(generate(messages))

question = "How many barbecues did I have in Summer 2019?"
question = "How long did I spend in Paris in 2019?"
get_recommendation(question)

#|%%--%%| <lJGUaxXoc6|Ii8H62jprp>
