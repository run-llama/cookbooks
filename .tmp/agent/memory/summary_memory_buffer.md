---
layout: recipe
colab: https://colab.research.google.com/github/run-llama/cookbooks-demo/blob/main/notebooks/agent/memory/summary_memory_buffer.ipynb
toc: True
title: "Chat Summary Memory Buffer"
featured: False
experimental: False
tags: ['Agent', 'Memory']
language: py
---
**NOTE:** This example of memory is deprecated in favor of the newer and more flexible `Memory` class. See the [latest docs](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/memory/).

The `ChatSummaryMemoryBuffer` is a memory buffer that stores the last X messages that fit into a token limit. It also summarizes the chat history into a single message.



```python
%pip install llama-index-core
```

## Setup


```python
from llama_index.core.memory import ChatSummaryMemoryBuffer

memory = ChatSummaryMemoryBuffer.from_defaults(
    token_limit=40000,
    # optional set the summary prompt, here's the default:
    # summarize_prompt=(
    #     "The following is a conversation between the user and assistant. "
    #     "Write a concise summary about the contents of this conversation."
    # )
)
```

## Using Standalone


```python
from llama_index.core.llms import ChatMessage

chat_history = [
    ChatMessage(role="user", content="Hello, how are you?"),
    ChatMessage(role="assistant", content="I'm doing well, thank you!"),
]

# put a list of messages
memory.put_messages(chat_history)

# put one message at a time
# memory.put_message(chat_history[0])
```


```python
# Get the last X messages that fit into a token limit
history = memory.get()
```


```python
# Get all messages
all_history = memory.get_all()
```


```python
# clear the memory
memory.reset()
```

## Using with Agents

You can set the memory in any agent in the `.run()` method.


```python
import os

os.environ["OPENAI_API_KEY"] = "sk-proj-..."
```


```python
from llama_index.core.agent.workflow import ReActAgent, FunctionAgent
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI


memory = ChatMemoryBuffer.from_defaults(token_limit=40000)

agent = FunctionAgent(tools=[], llm=OpenAI(model="gpt-4o-mini"))

# context to hold the chat history/state
ctx = Context(agent)
```


```python
resp = await agent.run("Hello, how are you?", ctx=ctx, memory=memory)
```


```python
print(memory.get_all())
```

    [ChatMessage(role=<MessageRole.USER: 'user'>, additional_kwargs={}, blocks=[TextBlock(block_type='text', text='Hello, how are you?')]), ChatMessage(role=<MessageRole.ASSISTANT: 'assistant'>, additional_kwargs={}, blocks=[TextBlock(block_type='text', text="Hello! I'm just a program, so I don't have feelings, but I'm here and ready to help you. How can I assist you today?")])]

