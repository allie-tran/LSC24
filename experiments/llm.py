# auto-reload

# %load_ext autoreload
# %autoreload 2

import asyncio
import json
import os
from typing import List, Optional, Union

import requests
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from partialjson.json_parser import JSONParser
from pydantic import BaseModel
from rich import print

# |%%--%%| <ncdWWwxOEY|wpPH2fUB2F>
r"""°°°
# Load OpenAI API key and model name
Specify the OpenAI API key and model name to use for the completion in your '.env' file or your environment variables.
This makes use of the asynchronous OpenAI API client to generate completions.
°°°"""
# |%%--%%| <wpPH2fUB2F|cE58yjx8NV>

load_dotenv()

# Set up ChatGPT generation model
OPENAI_API = os.environ.get("OPENAI_API", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")


class LLM:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=OPENAI_API)

    async def generate(self, messages: List[ChatCompletionMessageParam]):
        request = await self.client.chat.completions.create(
            model=MODEL_NAME, messages=messages, stream=True
        )  # type: ignore
        async for chunk in request:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def get_template_messages(self) -> List[ChatCompletionMessageParam]:
        message = ChatCompletionSystemMessageParam(
            role="system",
            content="You are a helpful assistant to help me answer questions in a lifelog dataset. I will give you information and the system constraints and you will suggest me how to do it.",
        )
        return [message]


# |%%--%%| <cE58yjx8NV|EESjIjWFya>
r"""°°°
# Creating a prompt for the user to provide the necessary parameters for the lifelog retrieval system
This is an experimental idea that involves the LLM to set certain parameters for a question answering pipeline. All the parameters CAN be adjusted in the interface. This is just a suggestion.
°°°"""
# |%%--%%| <EESjIjWFya|gydPjH7UlF>
# This is still very long!!! TODO!
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
        "aggregate": [a way to aggregate the events, one of the following: "sum", "average, "max", "min"]
        }}
}}
No explanation is needed. Just provide the JSON structure.
```
"""
# |%%--%%| <gydPjH7UlF|LylJhTd6o5>
r"""°°°
# Test the LLM prompt with a question
Some questions are very straightforward and can be answered with a simple query. Others require more complex processing. This is a test to see how the LLM can help with the process.
°°°"""
# |%%--%%| <LylJhTd6o5|LKWdSuWDGV>

# Load the model and JSON parser
llm = LLM()
parser = JSONParser()


# Async function to generate completions
async def generate(messages: List[ChatCompletionMessageParam]):
    PROMPT_START_FLAG = "```json"
    text = ""
    async for response in llm.generate(messages):
        text += response
        if PROMPT_START_FLAG in text:
            try:
                parameters = parser.parse(text.split(PROMPT_START_FLAG)[1])
                yield parameters
            except Exception:
                continue


# Get the completion
async def get_recommendation(question: str):
    formatted_prompt = prompt.format(question=question)
    messages = llm.get_template_messages()
    messages.append(
        ChatCompletionUserMessageParam(role="user", content=formatted_prompt)
    )
    parameters = {}
    async for parameters in generate(messages):
        print(end="\033c", flush=True)
        print(json.dumps(parameters))
    return parameters


# Run the function
question = "How many pints of Guinness are consumed on St. Patrick's Day in 2023?"
asyncio.run(get_recommendation(question))
print()
# |%%--%%| <LKWdSuWDGV|mabyKntgXt>
r"""°°°
# Let's try with ElasticSearch/current MySceal API
We can use the MySceal API to get the data we need. This is a test to see if we can get the data we need from the API.
°°°"""
# |%%--%%| <mabyKntgXt|R1XlacbsvN>
r"""°°°
## Let's reuse the MySceal API to retrieve the events now
°°°"""
# |%%--%%| <R1XlacbsvN|L7AS55kBw4>
# Define MySceal API endpoint
MYSCEAL_ENDPOINT = os.environ.get("MYSCEAL_ENDPOINT", "")


class MyScealQueryObject(BaseModel):
    current: str
    before: Optional[str] = ""
    beforewhen: Optional[str] = ""
    after: Optional[str] = ""
    afterwhen: Optional[str] = ""


# MySceal can take a query and return top-K results
class MyScealRequest(BaseModel):
    query: MyScealQueryObject
    gps_bounds: Optional[list[float]] = None
    size: Optional[int] = 100
    is_question: Optional[bool] = False
    last_scroll_id: Optional[str] = None


class MyScealImageResponse(BaseModel):
    image: str
    time: Union[str, int]
    shown: bool


class MyScealEventResponse(BaseModel):
    images: List[MyScealImageResponse]


def get_events_from_query(request: MyScealRequest) -> List[MyScealEventResponse]:
    """
    Getting list of top-100 images based on a query
    """
    PARAMS = {
        "query": {
            **request.query.__dict__,
            "isQuestion": request.is_question,
            "info": None,
        },
        "gps_bounds": None,
        "starting_from": 0,
        "share_info": False,
        "size": request.size,
    }

    response = requests.post(
        MYSCEAL_ENDPOINT + "/images/",
        data=json.dumps(PARAMS),
        headers={"Content-Type": "application/json"},
    )
    data = response.json()

    events = [triplet["current"] for triplet in data["results"]]
    events = [
        MyScealEventResponse(
            images=[
                MyScealImageResponse(image=image[0], time=image[1], shown=image[2])
                for image in event
            ]
        )
        for event in events
    ]
    return events


def get_more():
    """
    Getting list of top-100 images based on a query
    """
    PARAMS = {}
    response = requests.post(
        MYSCEAL_ENDPOINT + "/more/",
        data=json.dumps(PARAMS),
        headers={"Content-Type": "application/json"},
    )
    data = response.json()
    events = [triplet["current"] for triplet in data["results"]]
    events = [
        MyScealEventResponse(
            images=[
                MyScealImageResponse(image=image[0], time=image[1], shown=image[2])
                for image in event
            ]
        )
        for event in events
    ]
    return events


# |%%--%%| <L7AS55kBw4|rtdhmyjA4Q>
r"""°°°
### Test the MySceal API with a query
Let's test the MySceal API with a query to see if we can get the data we need.

°°°"""
# |%%--%%| <rtdhmyjA4Q|kPtV73Cull>

text_query = "Guinness pints on St. Patrick's Day in 2019"
request = MyScealRequest(query=MyScealQueryObject(current=text_query))

events = get_events_from_query(request)
for event in events[:5]:
    print("-" * 30)
    for image in event.images:
        if image.image:
            if image.shown:
                color = "bold green"
            else:
                color = "default"
            print(f"[{color}]{image.image}[/{color}]")
print("Yay it works!")


# |%%--%%| <kPtV73Cull|4DwnLP8heo>
r"""°°°
### Let's try to get some captions from the images
This idea is inspired by
[pan2023retrieving] Pan, Junting and Lin, Ziyi and Ge, Yuying and Zhu, Xiatian and Zhang, Renrui and Wang, Yi and Qiao, Yu and Li, Hongsheng "Retrieving-to-Answer: Zero-Shot Video Question Answering with FrozenLarge Language Models" (2023)

The idea is to retrieve captions from an uncurated dataset instead of generating captions. This is a test to see if we can get the captions from the images.

Let's try to see what dataset we can use for this.

°°°"""
# |%%--%%| <4DwnLP8heo|8wJk12RWwf>

# Get datasets from HuggingFace
wiki = load_dataset(
    "wikipedia", "20220301.en"
)  # this is 20.3GB in size and will take a while to download

# Inspect the dataset
print(wiki["train"][0])

# We are only interested in each sentence. Let's extract that.
for i, example in enumerate(wiki["train"]):
    print(example["text"])
    if i > 5:
        break

# Let's extract the sentences and save them to a file, so we can use them later. Beware of the memory usage!
with open("wiki_sentences.txt", "w") as f:
    for example in wiki["train"]:
        f.write(example["text"] + "\n")
