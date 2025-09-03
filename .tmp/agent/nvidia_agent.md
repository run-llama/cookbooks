---
layout: recipe
colab: https://colab.research.google.com/github/run-llama/cookbooks-demo/blob/main/notebooks/agent/nvidia_agent.ipynb
toc: True
title: "Function Calling NVIDIA Agent"
featured: False
experimental: False
tags: ['Agent', 'Integrations']
language: py
---
This notebook shows you how to use our NVIDIA agent, powered by function calling capabilities.

## Initial Setup 

Let's start by importing some simple building blocks.  

The main thing we need is:
1. the NVIDIA NIM Endpoint (using our own `llama_index` LLM class)
2. a place to keep conversation history 
3. a definition for tools that our agent can use.


```python
%pip install --upgrade --quiet llama-index-llms-nvidia
```


```python
import getpass
import os

# del os.environ['NVIDIA_API_KEY']  ## delete key and reset
if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    print("Valid NVIDIA_API_KEY already in environment. Delete to reset")
else:
    nvapi_key = getpass.getpass("NVAPI Key (starts with nvapi-): ")
    assert nvapi_key.startswith(
        "nvapi-"
    ), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key
```

    Valid NVIDIA_API_KEY already in environment. Delete to reset



```python
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.nvidia import NVIDIAEmbedding
```

Let's define some very simple calculator tools for our agent.


```python
def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b
```

Here we initialize a simple NVIDIA agent with calculator functions.


```python
llm = NVIDIA("meta/llama-3.1-70b-instruct")
```


```python
from llama_index.core.agent.workflow import FunctionAgent

agent = FunctionAgent(
    tools=[multiply, add],
    llm=llm,
)
```

### Chat


```python
response = await agent.run("What is (121 * 3) + 42?")
print(str(response))
```


```python
# inspect sources
print(response.tool_calls)
```

### Managing Context/Memory

By default, `.run()` is stateless. If you want to maintain state, you can pass in a `context` object.


```python
from llama_index.core.agent.workflow import Context

ctx = Context(agent)

response = await agent.run("Hello, my name is John Doe.", ctx=ctx)
print(str(response))

response = await agent.run("What is my name?", ctx=ctx)
print(str(response))
```

### Agent with Personality

You can specify a system prompt to give the agent additional instruction or personality.


```python
agent = FunctionAgent(
    tools=[multiply, add],
    llm=llm,
    system_prompt="Talk like a pirate in every response.",
)
```


```python
response = await agent.run("Hi")
print(response)
```


```python
response = await agent.run("Tell me a story")
print(response)
```

# NVIDIA Agent with RAG/Query Engine Tools


```python
!mkdir -p 'data/10k/'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
```


```python
from llama_index.core.tools import QueryEngineTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

# load data
uber_docs = SimpleDirectoryReader(
    input_files=["./data/10k/uber_2021.pdf"]
).load_data()

# build index
uber_index = VectorStoreIndex.from_documents(
    uber_docs, embed_model=embed_model
)
uber_engine = uber_index.as_query_engine(similarity_top_k=3, llm=llm)
query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=uber_engine,
    name="uber_10k",
    description=(
        "Provides information about Uber financials for year 2021. "
        "Use a detailed plain text question as input to the tool."
    ),
)
```


```python
agent = FunctionAgent(tools=[query_engine_tool], llm=llm)
```


```python
response = await agent.run(
    "Tell me both the risk factors and tailwinds for Uber? Do two parallel tool calls."
)
print(str(response))
```

# ReAct Agent 


```python
from llama_index.core.agent.workflow import ReActAgent
```


```python
agent = ReActAgent([multiply_tool, add_tool], llm=llm, verbose=True)
```

Using the `stream_events()` method, we can stream the response as it is generated to see the agent's thought process.

The final response will have only the final answer.


```python
from llama_index.core.agent.workflow import AgentStream

handler = agent.run("What is 20+(2*4)? Calculate step by step ")
async for ev in handler.stream_events():
    if isinstance(ev, AgentStream):
        print(ev.delta, end="", flush=True)

response = await handler
```


```python
print(str(response))
```


```python
print(response.tool_calls)
```
