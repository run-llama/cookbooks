---
layout: recipe
colab: https://colab.research.google.com/github/run-llama/cookbooks-demo/blob/main/notebooks/agent/memory/vector_memory.ipynb
toc: True
title: "Vector Memory"
featured: False
experimental: False
tags: ['Agent', 'Memory']
language: py
---
**NOTE:** This example of memory is deprecated in favor of the newer and more flexible `Memory` class. See the [latest docs](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/memory/).

The vector memory module uses vector search (backed by a vector db) to retrieve relevant conversation items given a user input.

This notebook shows you how to use the `VectorMemory` class. We show you how to use its individual functions. A typical usecase for vector memory is as a long-term memory storage of chat messages. You can

![VectorMemoryIllustration](https://d3ddy8balm3goa.cloudfront.net/llamaindex/vector-memory.excalidraw.svg)

### Initialize and Experiment with Memory Module

Here we initialize a raw memory module and demonstrate its functions - to put and retrieve from ChatMessage objects.

- Note that `retriever_kwargs` is the same args you'd specify on the `VectorIndexRetriever` or from `index.as_retriever(..)`.


```python
from llama_index.core.memory import VectorMemory
from llama_index.embeddings.openai import OpenAIEmbedding


vector_memory = VectorMemory.from_defaults(
    vector_store=None,  # leave as None to use default in-memory vector store
    embed_model=OpenAIEmbedding(),
    retriever_kwargs={"similarity_top_k": 1},
)
```


```python
from llama_index.core.llms import ChatMessage

msgs = [
    ChatMessage.from_str("Jerry likes juice.", "user"),
    ChatMessage.from_str("Bob likes burgers.", "user"),
    ChatMessage.from_str("Alice likes apples.", "user"),
]
```


```python
# load into memory
for m in msgs:
    vector_memory.put(m)
```


```python
# retrieve from memory
msgs = vector_memory.get("What does Jerry like?")
msgs
```




    [ChatMessage(role=<MessageRole.USER: 'user'>, content='Jerry likes juice.', additional_kwargs={})]




```python
vector_memory.reset()
```

Now let's try resetting and trying again. This time, we'll add an assistant message. Note that user/assistant messages are bundled by default.


```python
msgs = [
    ChatMessage.from_str("Jerry likes burgers.", "user"),
    ChatMessage.from_str("Bob likes apples.", "user"),
    ChatMessage.from_str("Indeed, Bob likes apples.", "assistant"),
    ChatMessage.from_str("Alice likes juice.", "user"),
]
vector_memory.set(msgs)
```


```python
msgs = vector_memory.get("What does Bob like?")
msgs
```




    [ChatMessage(role=<MessageRole.USER: 'user'>, content='Bob likes apples.', additional_kwargs={}),
     ChatMessage(role=<MessageRole.ASSISTANT: 'assistant'>, content='Indeed, Bob likes apples.', additional_kwargs={})]


