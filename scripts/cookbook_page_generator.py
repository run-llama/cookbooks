#!/usr/bin/env python3
"""
Cookbook Documentation Generator

Reads index.toml and generates an HTML documentation page with filtering and search.
"""

import tomllib
from pathlib import Path
from typing import List, Dict, Set
import json
from urllib.parse import urljoin

class CookbookGenerator:
    def __init__(self, toml_path: str, output_path: str = "cookbooks.html"):
        self.toml_path = toml_path
        self.output_path = output_path
        self.data = None
        
    def load_toml(self):
        """Load and parse the index.toml file"""
        with open(self.toml_path, 'rb') as f:
            self.data = tomllib.load(f)
    
    def get_all_tags(self) -> Set[str]:
        """Extract all unique tags from recipes"""
        tags = set()
        for recipe in self.data.get('recipe', []):
            tags.update(recipe.get('tags', []))
        return sorted(tags)
    
    def get_all_languages(self) -> Set[str]:
        """Extract all unique languages from recipes"""
        languages = set()
        for recipe in self.data.get('recipe', []):
            lang = recipe.get('language', '')
            if lang:
                languages.add(lang)
        return sorted(languages)
    
    def generate_notebook_url(self, recipe: Dict) -> str:
        """Generate the notebook URL based on config and recipe data"""
        config = self.data.get('config', {})
        colab_base = config.get('colab', '')
        
        if 'notebook' in recipe:
            return urljoin(colab_base, recipe['notebook'])
        elif 'source' in recipe:
            return recipe['source']
        return "#"
    
    def generate_html(self) -> str:
        """Generate the complete HTML page"""
        if not self.data:
            raise ValueError("No data loaded. Call load_toml() first.")
        
        recipes = self.data.get('recipe', [])
        all_tags = self.get_all_tags()
        all_languages = self.get_all_languages()
        
        # Convert recipes to JSON for JavaScript
        recipes_json = json.dumps([
            {
                'title': recipe.get('title', ''),
                'description': recipe.get('description', ''),
                'tags': recipe.get('tags', []),
                'language': recipe.get('language', ''),
                'url': self.generate_notebook_url(recipe),
                'featured': recipe.get('featured', False),
                'experimental': recipe.get('experimental', False)
            }
            for recipe in recipes
        ])
        
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cookbooks Documentation</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f8f9fa;
            color: #333;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: white;
            padding: 40px 20px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .page-title {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            color: #1a1a1a;
        }}
        
        .page-subtitle {{
            font-size: 1.1rem;
            color: #666;
            max-width: 600px;
            margin: 0 auto;
        }}
        
        .controls-section {{
            background: white;
            padding: 30px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .search-filters-row {{
            display: flex;
            gap: 20px;
            align-items: flex-end;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }}
        
        .search-container {{
            flex: 1;
            min-width: 300px;
        }}
        
        .filter-group {{
            min-width: 150px;
        }}
        
        .label {{
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 0.9rem;
            color: #374151;
        }}
        
        .search-input, .filter-select {{
            width: 100%;
            padding: 12px;
            border: 2px solid #e5e7eb;
            border-radius: 6px;
            font-size: 16px;
            transition: border-color 0.2s;
        }}
        
        .search-input:focus, .filter-select:focus {{
            outline: none;
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }}
        
        .active-filters {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
        }}
        
        .filter-tag {{
            background: #3b82f6;
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .filter-tag .remove {{
            cursor: pointer;
            font-weight: bold;
            padding: 2px;
        }}
        
        .clear-all {{
            background: #ef4444;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
        }}
        
        .results-info {{
            background: #f0f9ff;
            border: 1px solid #0ea5e9;
            border-left: 4px solid #0ea5e9;
            padding: 12px 20px;
            margin-bottom: 20px;
            border-radius: 4px;
            font-size: 0.9rem;
            color: #0c4a6e;
        }}
        
        .cookbooks-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 24px;
        }}
        
        .cookbook-card {{
            background: white;
            border-radius: 8px;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: all 0.2s;
            position: relative;
            border: 2px solid transparent;
        }}
        
        .cookbook-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
            border-color: #3b82f6;
        }}
        
        .cookbook-card.featured {{
            border-color: #f59e0b;
            background: linear-gradient(135deg, #fefbf3 0%, #ffffff 100%);
        }}
        
        .cookbook-card.experimental {{
            border-color: #8b5cf6;
            background: linear-gradient(135deg, #faf5ff 0%, #ffffff 100%);
        }}
        
        .featured-badge, .experimental-badge {{
            position: absolute;
            top: -8px;
            right: 16px;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .featured-badge {{
            background: #f59e0b;
            color: white;
        }}
        
        .experimental-badge {{
            background: #8b5cf6;
            color: white;
        }}
        
        .cookbook-title {{
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 12px;
            color: #1a1a1a;
        }}
        
        .cookbook-description {{
            color: #6b7280;
            margin-bottom: 16px;
            line-height: 1.5;
        }}
        
        .cookbook-tags {{
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }}
        
        .tag {{
            background: #f3f4f6;
            color: #374151;
            padding: 4px 10px;
            border-radius: 16px;
            font-size: 0.8rem;
            border: 1px solid #e5e7eb;
        }}
        
        .cookbook-footer {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .cookbook-link {{
            color: #3b82f6;
            text-decoration: none;
            font-weight: 500;
            font-size: 0.9rem;
        }}
        
        .cookbook-link:hover {{
            text-decoration: underline;
        }}
        
        .language-badge {{
            background: #1f2937;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            text-transform: uppercase;
        }}
        
        .empty-state {{
            text-align: center;
            padding: 60px 20px;
            color: #6b7280;
        }}
        
        .empty-state-icon {{
            font-size: 3rem;
            margin-bottom: 16px;
        }}
        
        @media (max-width: 768px) {{
            .search-filters-row {{
                flex-direction: column;
                align-items: stretch;
            }}
            
            .cookbooks-grid {{
                grid-template-columns: 1fr;
            }}
            
            .cookbook-card {{
                padding: 20px;
            }}
            
            .page-title {{
                font-size: 2rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1 class="page-title">Cookbooks Documentation</h1>
            <p class="page-subtitle">Explore our collection of practical guides and tutorials for building AI applications</p>
        </div>

        <!-- Search and Filter Controls -->
        <div class="controls-section">
            <div class="search-filters-row">
                <div class="search-container">
                    <label class="label" for="search">Search Cookbooks</label>
                    <input type="text" id="search" class="search-input" placeholder="Search by title, description, or tags...">
                </div>
                
                <div class="filter-group">
                    <label class="label" for="tag-filter">Filter by Tag</label>
                    <select id="tag-filter" class="filter-select">
                        <option value="">All Tags</option>
                        {''.join(f'<option value="{tag}">{tag}</option>' for tag in all_tags)}
                    </select>
                </div>
                
                <div class="filter-group">
                    <label class="label" for="language-filter">Language</label>
                    <select id="language-filter" class="filter-select">
                        <option value="">All Languages</option>
                        {''.join(f'<option value="{lang}">{lang.upper()}</option>' for lang in all_languages)}
                    </select>
                </div>
                
                <div class="filter-group">
                    <label class="label" for="sort-select">Sort By</label>
                    <select id="sort-select" class="filter-select">
                        <option value="title">Title A-Z</option>
                        <option value="featured">Featured First</option>
                        <option value="tags">By Tag</option>
                    </select>
                </div>
            </div>
            
            <div class="active-filters" id="active-filters">
                <!-- Active filters will be populated by JavaScript -->
            </div>
        </div>

        <!-- Results Info -->
        <div class="results-info" id="results-info">
            Loading cookbooks...
        </div>

        <!-- Cookbooks Grid -->
        <div class="cookbooks-grid" id="cookbooks-grid">
            <!-- Cookbook cards will be populated by JavaScript -->
        </div>
        
        <!-- Empty State -->
        <div class="empty-state" id="empty-state" style="display: none;">
            <div class="empty-state-icon">📚</div>
            <h3>No cookbooks found</h3>
            <p>Try adjusting your search or filter criteria</p>
        </div>
    </div>

    <script>
        // Cookbook data from TOML
        const cookbooks = {recipes_json};
        let filteredCookbooks = [...cookbooks];
        
        // DOM elements
        const searchInput = document.getElementById('search');
        const tagFilter = document.getElementById('tag-filter');
        const languageFilter = document.getElementById('language-filter');
        const sortSelect = document.getElementById('sort-select');
        const resultsInfo = document.getElementById('results-info');
        const cookbooksGrid = document.getElementById('cookbooks-grid');
        const emptyState = document.getElementById('empty-state');
        const activeFiltersContainer = document.getElementById('active-filters');
        
        function createCookbookCard(cookbook) {{
            const badgeHtml = cookbook.featured 
                ? '<div class="featured-badge">Featured</div>' 
                : cookbook.experimental 
                    ? '<div class="experimental-badge">Experimental</div>'
                    : '';
            
            const cardClass = cookbook.featured 
                ? 'cookbook-card featured' 
                : cookbook.experimental 
                    ? 'cookbook-card experimental'
                    : 'cookbook-card';
            
            const tagsHtml = cookbook.tags.map(tag => 
                `<span class="tag">${{tag}}</span>`
            ).join('');
            
            const languageBadge = cookbook.language 
                ? `<span class="language-badge">${{cookbook.language}}</span>`
                : '';
            
            return `
                <div class="${{cardClass}}">
                    ${{badgeHtml}}
                    <div class="cookbook-title">${{cookbook.title}}</div>
                    <div class="cookbook-description">${{cookbook.description}}</div>
                    <div class="cookbook-tags">${{tagsHtml}}</div>
                    <div class="cookbook-footer">
                        <a href="${{cookbook.url}}" class="cookbook-link" target="_blank">
                            Open Cookbook →
                        </a>
                        ${{languageBadge}}
                    </div>
                </div>
            `;
        }}
        
        function updateActiveFilters() {{
            const activeFilters = [];
            
            if (tagFilter.value) {{
                activeFilters.push({{
                    type: 'tag',
                    value: tagFilter.value,
                    label: `Tag: ${{tagFilter.value}}`
                }});
            }}
            
            if (languageFilter.value) {{
                activeFilters.push({{
                    type: 'language',
                    value: languageFilter.value,
                    label: `Language: ${{languageFilter.value.toUpperCase()}}`
                }});
            }}
            
            if (searchInput.value.trim()) {{
                activeFilters.push({{
                    type: 'search',
                    value: searchInput.value,
                    label: `Search: "${{searchInput.value}}"`
                }});
            }}
            
            const filtersHtml = activeFilters.map(filter => `
                <div class="filter-tag">
                    ${{filter.label}}
                    <span class="remove" onclick="removeFilter('${{filter.type}}', '${{filter.value}}')">×</span>
                </div>
            `).join('');
            
            if (activeFilters.length > 0) {{
                activeFiltersContainer.innerHTML = filtersHtml + 
                    '<button class="clear-all" onclick="clearAllFilters()">Clear All</button>';
            }} else {{
                activeFiltersContainer.innerHTML = '';
            }}
        }}
        
        function removeFilter(type, value) {{
            if (type === 'tag') {{
                tagFilter.value = '';
            }} else if (type === 'language') {{
                languageFilter.value = '';
            }} else if (type === 'search') {{
                searchInput.value = '';
            }}
            filterAndRender();
        }}
        
        function clearAllFilters() {{
            searchInput.value = '';
            tagFilter.value = '';
            languageFilter.value = '';
            sortSelect.value = 'title';
            filterAndRender();
        }}
        
        function filterAndRender() {{
            const searchTerm = searchInput.value.toLowerCase().trim();
            const selectedTag = tagFilter.value;
            const selectedLanguage = languageFilter.value;
            const sortBy = sortSelect.value;
            
            // Filter cookbooks
            filteredCookbooks = cookbooks.filter(cookbook => {{
                // Search filter
                const matchesSearch = !searchTerm || 
                    cookbook.title.toLowerCase().includes(searchTerm) ||
                    cookbook.description.toLowerCase().includes(searchTerm) ||
                    cookbook.tags.some(tag => tag.toLowerCase().includes(searchTerm));
                
                // Tag filter
                const matchesTag = !selectedTag || cookbook.tags.includes(selectedTag);
                
                // Language filter
                const matchesLanguage = !selectedLanguage || cookbook.language === selectedLanguage;
                
                return matchesSearch && matchesTag && matchesLanguage;
            }});
            
            // Sort cookbooks
            filteredCookbooks.sort((a, b) => {{
                if (sortBy === 'featured') {{
                    if (a.featured && !b.featured) return -1;
                    if (!a.featured && b.featured) return 1;
                    if (a.experimental && !b.experimental) return -1;
                    if (!a.experimental && b.experimental) return 1;
                }}
                if (sortBy === 'tags') {{
                    return a.tags[0]?.localeCompare(b.tags[0] || '') || 0;
                }}
                return a.title.localeCompare(b.title);
            }});
            
            // Update results info
            resultsInfo.textContent = `Showing ${{filteredCookbooks.length}} of ${{cookbooks.length}} cookbooks`;
            
            // Update active filters display
            updateActiveFilters();
            
            // Render results
            if (filteredCookbooks.length === 0) {{
                cookbooksGrid.style.display = 'none';
                emptyState.style.display = 'block';
            }} else {{
                cookbooksGrid.style.display = 'grid';
                emptyState.style.display = 'none';
                cookbooksGrid.innerHTML = filteredCookbooks.map(createCookbookCard).join('');
            }}
        }}
        
        // Event listeners
        searchInput.addEventListener('input', filterAndRender);
        tagFilter.addEventListener('change', filterAndRender);
        languageFilter.addEventListener('change', filterAndRender);
        sortSelect.addEventListener('change', filterAndRender);
        
        // Initialize page
        filterAndRender();
        
        // Add some interactivity hints
        console.log('Cookbook Documentation Page Loaded');
        console.log(`Total cookbooks: ${{cookbooks.length}}`);
        console.log(`Available tags: {', '.join(all_tags)}`);
    </script>
</body>
</html>"""
        
        return html_template
    
    def generate_file(self):
        """Generate the HTML file"""
        self.load_toml()
        html_content = self.generate_html()
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Generated cookbook documentation: {self.output_path}")
        print(f"Total recipes: {len(self.data.get('recipe', []))}")
        print(f"Unique tags: {', '.join(self.get_all_tags())}")

def main():
    """Main function to run the generator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate cookbook documentation from index.toml')
    parser.add_argument('toml_file', help='Path to index.toml file')
    parser.add_argument('-o', '--output', default='cookbooks.html', 
                       help='Output HTML file (default: cookbooks.html)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.toml_file).exists():
        print(f"Error: File {args.toml_file} not found")
        return 1
    
    try:
        generator = CookbookGenerator(args.toml_file, args.output)
        generator.generate_file()
        return 0
    except Exception as e:
        print(f"Error generating documentation: {e}")
        return 1

if __name__ == "__main__":
    exit(main())