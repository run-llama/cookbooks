# Cookbook POC

### The Idea for Pyhton Examples


1. You add a notebook wherever you want in the `/notebooks` directory OR, it already exists in another `source`
2. You want it to go onto the website? You add an entry to "index.toml" for your example. Here's the info we need:
    - A description
    - The location of your example
    - Some tags 
    - If you want, you can mark it 'experimental'
    - If _we_ want, we can make it 'featured'
    - Mark the `language` as `language = "py"`
3. A github action on this repo or on the `developers` repo runs the `scripts/notebooks_to_markdown.py` script. Which converts your notebook to markdown with some frontmatter which includes:
    - all of the info above
    - an auto generate 'open in Colab' url so you don't have to worry about adding it yourself
4. (This is my idea but we can change it) In Astro, we use this frontmatter to generate a tiled 'Cookbook' page with tags and filters. You can easilyl navigate different topics and all LlamaCloud/LITS/LI examples. When you click-> takes you to usual Examples page.

> Extra: This POC has most examples as a local `notebook` and one example where the recipe comes from another `source`. For those that come from a local `notebook`, we also generate a frontmatter element called `colab` which auto-generates the 'open in colab' url. This can be used for an 'open in colab' button 🚀

### (Optional) Example Generated frontmatter for Astro to generate individual recipe pages:
This is useful if you want to use the generated frontmatter which can be used to add elements to individual recipes like tags, an 'open in colab' button, etc.

```
---
layout: recipe
colab: https://colab.research.google.com/github/run-llama/cookbooks-demo/blob/main/notebooks/agent/agent_builder.ipynb
toc: True
title: "GPT Builder Demo"
featured: False
experimental: False
tags: ['Agent']
language: py
---
```
### Example build run for the astro website:

```bash
pip install -r requirements.txt
python scripts/notebooks_to_markdown.py
```

This adds all of the _local_ notebooks as markdown, into the `makrdowns` directory, along with their frontmatter.


### POC Landing Page Generation for Astro

> Disclaimer: This index.toml to landing page generation code was created with Claude :) 

```bash
python scripts/cookbook_page_generator.py index.toml -o my_cookbooks.html
```
This outputs a POC HTML page that you can have a look at. It's just to demonstrate how the index.toml can be used. 
Note that it has both local notebooks + external sources