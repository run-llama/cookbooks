# Cookbook POC

### The Idea for Pyhton Examples

1. You add a notebook wherever you want in the `/notebooks` directory
2. You want it to go onto the website? You add an entry to "index.toml" for your example. Here's the info we need:
    - A description
    - The location of your example
    - Some tags 
    - If you want, you can mark it 'experimental'
    - If _we_ want, we can make it 'featured'
3. A github action on this repo or on the `developers` repo runs the `scripts/notebooks_to_markdown.py` script. Which converts your notebook to markdown with some frontmatter which includes:
    - all of the info above
    - an auto generate 'open in Colab' url so you don't have to worry about adding it yourself
4. (This is my idea but we can change it) In Astro, we use this frontmatter to generate a tiled 'Cookbook' page with tags and filters. You can easilyl navigate different topics and all LlamaCloud/LITS/LI examples. When you click-> takes you to usual Examples page.