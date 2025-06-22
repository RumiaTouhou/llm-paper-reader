#!/usr/bin/env python3
"""
MarkdownBrowser - Direct browser-based markdown viewer with LaTeX support

Usage:
    python MarkdownBrowser.py              # Auto-loads SamplePaper.md
    python MarkdownBrowser.py document.md  # Loads specific file

As a module:
    from MarkdownBrowser import DirectMarkdownBrowser, Plugin, create_browser
    
    browser = create_browser("document.md")
    browser.run()
"""

import sys
import os
import markdown
import webbrowser
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
import urllib.parse

# Check for pymdown-extensions
LATEX_SUPPORT = False
try:
    import pymdownx
    LATEX_SUPPORT = True
except ImportError:
    print("Note: Install pymdown-extensions for better LaTeX parsing")

# Export main classes and functions for module usage
__all__ = ['DirectMarkdownBrowser', 'Plugin', 'PluginSystem', 'create_browser', 'create_formula_index_plugin']


@dataclass
class Plugin:
    """Plugin configuration for extending the browser"""
    name: str
    html_content: Optional[str] = None  # HTML to inject
    javascript: Optional[str] = None    # JS to inject
    css: Optional[str] = None          # CSS to inject
    markdown_preprocessor: Optional[Callable[[str], str]] = None
    html_postprocessor: Optional[Callable[[str], str]] = None
    api_endpoints: Optional[Dict[str, Callable]] = None  # API handlers


class PluginSystem:
    """Manages plugins for the browser"""
    
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self._preprocessors: List[Callable] = []
        self._postprocessors: List[Callable] = []
        
    def register(self, plugin: Plugin):
        """Register a new plugin"""
        self.plugins[plugin.name] = plugin
        
        if plugin.markdown_preprocessor:
            self._preprocessors.append(plugin.markdown_preprocessor)
        
        if plugin.html_postprocessor:
            self._postprocessors.append(plugin.html_postprocessor)
    
    def preprocess_markdown(self, text: str) -> str:
        """Apply all markdown preprocessors"""
        for processor in self._preprocessors:
            text = processor(text)
        return text
    
    def postprocess_html(self, html: str) -> str:
        """Apply all HTML postprocessors"""
        for processor in self._postprocessors:
            html = processor(html)
        return html
    
    def get_plugin_html(self) -> str:
        """Get combined HTML from all plugins"""
        html_parts = []
        for plugin in self.plugins.values():
            if plugin.html_content:
                html_parts.append(f'<div class="plugin-widget" data-plugin="{plugin.name}">')
                html_parts.append(plugin.html_content)
                html_parts.append('</div>')
        return '\n'.join(html_parts)
    
    def get_plugin_css(self) -> str:
        """Get combined CSS from all plugins"""
        css_parts = []
        for plugin in self.plugins.values():
            if plugin.css:
                css_parts.append(f'/* Plugin: {plugin.name} */')
                css_parts.append(plugin.css)
        return '\n'.join(css_parts)
    
    def get_plugin_javascript(self) -> str:
        """Get combined JavaScript from all plugins"""
        js_parts = []
        for plugin in self.plugins.values():
            if plugin.javascript:
                js_parts.append(f'// Plugin: {plugin.name}')
                js_parts.append(plugin.javascript)
        return '\n'.join(js_parts)


class MarkdownRenderer:
    """Converts markdown to HTML with LaTeX support"""
    
    def __init__(self, plugin_system: PluginSystem):
        self.plugin_system = plugin_system
        
        # Configure extensions
        extensions = ['extra', 'codehilite', 'toc', 'tables', 'fenced_code', 'attr_list']
        extension_configs = {
            'toc': {
                'title': 'Table of Contents', 
                'anchorlink': True,
                'anchorlink_class': 'toc-anchor',
                'permalink': False,
                'baselevel': 1,
                'toc_depth': '1-6'
            },
            'codehilite': {
                'css_class': 'highlight',
                'guess_lang': False
            }
        }
        
        if LATEX_SUPPORT:
            extensions.extend([
                'pymdownx.arithmatex',
                'pymdownx.superfences',
                'pymdownx.tasklist',
                'pymdownx.tilde',
            ])
            extension_configs['pymdownx.arithmatex'] = {
                'generic': True,
            }
        
        self.md = markdown.Markdown(
            extensions=extensions,
            extension_configs=extension_configs
        )
    
    def render(self, markdown_text: str) -> tuple:
        """Convert markdown to HTML with plugin processing"""
        markdown_text = self.plugin_system.preprocess_markdown(markdown_text)
        
        self.md.reset()
        html_content = self.md.convert(markdown_text)
        
        # Build TOC with proper IDs
        html_content, toc_html = self._process_headers_and_build_toc(html_content)
        
        html_content = self.plugin_system.postprocess_html(html_content)
        
        return html_content, toc_html
    
    def _process_headers_and_build_toc(self, html_content: str) -> tuple:
        """Add IDs to headers and build TOC"""
        import re
        from html import escape
        
        header_pattern = r'<h([1-6])([^>]*)>(.+?)</h\1>'
        headers = list(re.finditer(header_pattern, html_content, re.DOTALL))
        
        if not headers:
            return html_content, ''
        
        toc_items = []
        modified_html = html_content
        offset = 0
        
        for match in headers:
            level = int(match.group(1))
            attrs = match.group(2)
            content = match.group(3)
            
            # Clean text for ID
            clean_text = re.sub(r'<[^>]+>', '', content).strip()
            clean_text = re.sub(r'\s+', ' ', clean_text)
            
            # Get or generate ID
            id_match = re.search(r'id="([^"]+)"', attrs)
            if id_match:
                header_id = id_match.group(1)
            else:
                header_id = clean_text.lower()
                header_id = re.sub(r'[^\w\s-]', '', header_id)
                header_id = re.sub(r'[-\s]+', '-', header_id).strip('-')
                
                # Add ID to header
                new_header = f'<h{level}{attrs} id="{header_id}">{content}</h{level}>'
                start = match.start() + offset
                end = match.end() + offset
                modified_html = modified_html[:start] + new_header + modified_html[end:]
                offset += len(new_header) - len(match.group(0))
            
            toc_items.append({
                'level': level,
                'id': header_id,
                'text': clean_text
            })
        
        toc_html = self._build_toc_html(toc_items)
        return modified_html, toc_html
    
    def _build_toc_html(self, toc_items: list) -> str:
        """Build hierarchical TOC HTML"""
        if not toc_items:
            return ''
        
        from html import escape
        
        min_level = min(item['level'] for item in toc_items)
        html_parts = []
        stack = []  # (level, has_children)
        
        for i, item in enumerate(toc_items):
            level = item['level'] - min_level + 1
            
            # Close deeper levels
            while stack and stack[-1][0] >= level:
                html_parts.append('</li>')
                if stack[-1][1]:
                    html_parts.append('</ul>')
                stack.pop()
            
            # Open new levels
            while len(stack) < level - 1:
                html_parts.append('<ul>')
                html_parts.append('<li>')
                stack.append((len(stack) + 1, False))
            
            # Check for children
            has_children = (i + 1 < len(toc_items) and 
                          toc_items[i + 1]['level'] > item['level'])
            
            # Add item
            if stack and not stack[-1][1]:
                html_parts.append('</li>')
            
            html_parts.append('<li>')
            html_parts.append(f'<a href="#{item["id"]}">{escape(item["text"])}</a>')
            
            if has_children:
                html_parts.append('<ul>')
                stack.append((level, True))
            else:
                stack.append((level, False))
        
        # Close remaining
        while stack:
            html_parts.append('</li>')
            if stack[-1][1]:
                html_parts.append('</ul>')
            stack.pop()
        
        return '<ul>\n' + '\n'.join(html_parts) + '\n</ul>'


class BrowserHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the local server"""
    
    def __init__(self, *args, browser_instance=None, **kwargs):
        self.browser = browser_instance
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        """Suppress server logs"""
        pass
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.browser.get_html().encode())
        elif self.path == '/api/reload':
            # Reload the document
            self.browser.reload()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'reloaded'}).encode())
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests for plugin APIs"""
        if self.path.startswith('/api/plugin/'):
            # Extract plugin name and endpoint
            parts = self.path.split('/')
            if len(parts) >= 4:
                plugin_name = parts[3]
                endpoint = '/'.join(parts[4:]) if len(parts) > 4 else ''
                
                try:
                    # Get request data
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length).decode('utf-8')
                    
                    # Call plugin endpoint
                    plugin = self.browser.plugin_system.plugins.get(plugin_name)
                    if plugin and plugin.api_endpoints and endpoint in plugin.api_endpoints:
                        try:
                            data = json.loads(post_data) if post_data else {}
                            result = plugin.api_endpoints[endpoint](data)
                            
                            # Try to send response
                            try:
                                self.send_response(200)
                                self.send_header('Content-type', 'application/json')
                                self.end_headers()
                                self.wfile.write(json.dumps(result).encode())
                            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError) as e:
                                # Connection was closed by client, log but don't crash
                                print(f"[Browser] Client disconnected during response: {type(e).__name__}")
                                return
                                
                        except Exception as e:
                            # Handle errors in plugin execution
                            print(f"[Browser] Plugin error: {type(e).__name__}: {str(e)}")
                            try:
                                # Use ASCII-safe error message
                                self.send_error(500, "Internal Server Error")
                            except:
                                # If even error sending fails, just return
                                return
                    else:
                        self.send_error(404, "Not Found")
                except Exception as e:
                    # Handle any other errors
                    print(f"[Browser] Request handling error: {type(e).__name__}")
                    try:
                        self.send_error(400, "Bad Request")
                    except:
                        return
        else:
            self.send_error(404, "Not Found")


class DirectMarkdownBrowser:
    """Browser-based markdown viewer with plugin support"""
    
    def __init__(self, port=0):  # port=0 auto-selects
        self.plugin_system = PluginSystem()
        self.renderer = MarkdownRenderer(self.plugin_system)
        self.current_file = None
        self.current_content = ""
        self.server = None
        self.port = port
        self.server_thread = None
        
        self._register_core_plugins()
    
    def _register_core_plugins(self):
        """Register built-in plugins"""
        # Table of Contents plugin
        toc_plugin = Plugin(
            name="table-of-contents",
            html_content="""
<div class="toc-container">
    <h3>Table of Contents</h3>
    <div id="toc-content"></div>
</div>
""",
            css="""
.toc-container {
    position: fixed;
    left: 20px;
    top: 80px;
    width: 250px;
    max-height: 70vh;
    overflow-y: auto;
    background: white;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 15px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.toc-container h3 {
    margin-top: 0;
    font-size: 16px;
    border-bottom: 1px solid #eee;
    padding-bottom: 10px;
    margin-bottom: 10px;
}
#toc-content {
    font-size: 14px;
}
#toc-content ul {
    list-style: none;
    padding-left: 15px;
    margin: 5px 0;
}
#toc-content > ul {
    padding-left: 0;
}
#toc-content li {
    margin: 3px 0;
}
#toc-content a {
    text-decoration: none;
    color: #0366d6;
    display: block;
    padding: 2px 0;
}
#toc-content a:hover {
    text-decoration: underline;
    color: #0056b3;
}
/* Handle nested levels with indentation */
#toc-content ul ul {
    margin-top: 2px;
    border-left: 1px solid #eee;
    margin-left: 5px;
}
@media (max-width: 1200px) {
    .toc-container {
        display: none;
    }
}
""",
            javascript="""
document.addEventListener('DOMContentLoaded', function() {
    const tocElement = document.querySelector('.toc');
    const tocContent = document.getElementById('toc-content');
    const tocContainer = document.querySelector('.toc-container');
    
    if (tocElement && tocContent) {
        // Extract and display TOC
        tocContent.innerHTML = tocElement.innerHTML;
        tocElement.remove();
        
        // Add smooth scrolling
        tocContent.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const targetId = this.getAttribute('href').substring(1);
                const target = document.getElementById(targetId);
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    // Highlight briefly
                    target.style.backgroundColor = '#ffffcc';
                    target.style.transition = 'background-color 0.3s';
                    setTimeout(() => {
                        target.style.backgroundColor = '';
                    }, 2000);
                }
            });
        });
    } else if (tocContainer) {
        tocContainer.style.display = 'none';
    }
});
"""
        )
        self.register_plugin(toc_plugin)
        
        # Control panel plugin
        control_plugin = Plugin(
            name="controls",
            html_content="""
<div class="control-panel">
    <button onclick="reloadDocument()">🔄 Reload</button>
    <button onclick="toggleTheme()">🌓 Theme</button>
    <button onclick="printDocument()">🖨️ Print</button>
</div>
""",
            css="""
.control-panel {
    position: fixed;
    top: 20px;
    right: 20px;
    display: flex;
    gap: 10px;
    z-index: 1000;
}
.control-panel button {
    padding: 8px 15px;
    border: 1px solid #ddd;
    background: white;
    border-radius: 5px;
    cursor: pointer;
    font-size: 14px;
}
.control-panel button:hover {
    background: #f0f0f0;
}
body.dark-theme {
    background: #1a1a1a;
    color: #e0e0e0;
}
body.dark-theme .content-wrapper {
    background: #2a2a2a;
}
body.dark-theme .toc-container {
    background: #2a2a2a;
    border-color: #444;
}
body.dark-theme .control-panel button {
    background: #2a2a2a;
    color: #e0e0e0;
    border-color: #444;
}
""",
            javascript="""
function reloadDocument() {
    fetch('/api/reload')
        .then(() => location.reload());
}

function toggleTheme() {
    document.body.classList.toggle('dark-theme');
    localStorage.setItem('theme', document.body.classList.contains('dark-theme') ? 'dark' : 'light');
}

function printDocument() {
    window.print();
}

// Load saved theme
document.addEventListener('DOMContentLoaded', function() {
    if (localStorage.getItem('theme') === 'dark') {
        document.body.classList.add('dark-theme');
    }
});
"""
        )
        self.register_plugin(control_plugin)
    
    def register_plugin(self, plugin: Plugin):
        """Register a new plugin"""
        self.plugin_system.register(plugin)
    
    def get_html(self) -> str:
        """Generate complete HTML page"""
        html_content, toc_html = self.renderer.render(self.current_content)
        
        plugin_html = self.plugin_system.get_plugin_html()
        plugin_css = self.plugin_system.get_plugin_css()
        plugin_js = self.plugin_system.get_plugin_javascript()
        
        # Include hidden TOC for JavaScript extraction
        if toc_html:
            html_content = f'<div class="toc" style="display:none">{toc_html}</div>\n{html_content}'
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>MarkdownBrowser{' - ' + os.path.basename(self.current_file) if self.current_file else ''}</title>
    
    <!-- MathJax for LaTeX -->
    <script>
    window.MathJax = {{
        tex: {{
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
            processEscapes: true
        }},
        startup: {{
            pageReady: () => {{
                return MathJax.startup.defaultPageReady();
            }}
        }}
    }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }}
        
        .content-wrapper {{
            max-width: 900px;
            margin: 0 auto;
            padding: 40px;
            background: white;
            min-height: 100vh;
            box-shadow: 0 0 10px rgba(0,0,0,0.05);
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
        }}
        
        h1 {{ font-size: 2em; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }}
        h2 {{ font-size: 1.5em; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }}
        
        code {{
            background: #f6f8fa;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 85%;
        }}
        
        pre {{
            background: #f6f8fa;
            padding: 16px;
            overflow-x: auto;
            border-radius: 6px;
        }}
        
        pre code {{
            background: none;
            padding: 0;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
        }}
        
        th, td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }}
        
        th {{
            background: #f6f8fa;
            font-weight: 600;
        }}
        
        blockquote {{
            margin: 0;
            padding: 0 1em;
            color: #666;
            border-left: 4px solid #ddd;
        }}
        
        a {{
            color: #0366d6;
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        /* Hide original TOC */
        .toc {{
            display: none;
        }}
        
        /* Print styles */
        @media print {{
            .control-panel, .toc-container, .plugin-widget {{
                display: none !important;
            }}
            .content-wrapper {{
                box-shadow: none;
                padding: 0;
            }}
        }}
        
        /* Ensure headers have proper IDs for anchor links */
        h1[id], h2[id], h3[id], h4[id], h5[id], h6[id] {{
            scroll-margin-top: 20px;
        }}
        
        /* Add anchor link styling */
        .toc-anchor {{
            text-decoration: none;
            color: inherit;
        }}
        
        {plugin_css}
    </style>
</head>
<body>
    {plugin_html}
    
    <div class="content-wrapper">
        {html_content}
    </div>
    
    <script>
        {plugin_js}
    </script>
</body>
</html>"""
    
    def start_server(self):
        """Start the local HTTP server"""
        handler = lambda *args, **kwargs: BrowserHandler(*args, browser_instance=self, **kwargs)
        self.server = HTTPServer(('localhost', self.port), handler)
        self.port = self.server.server_port  # Get actual port if auto-selected
        
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        return self.port
    
    def stop_server(self):
        """Stop the HTTP server"""
        if self.server:
            self.server.shutdown()
            self.server_thread.join()
    
    def load_markdown_file(self, file_path: str):
        """Load and display a markdown file"""
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found")
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.current_content = f.read()
            
            self.current_file = file_path
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def load_markdown_content(self, content: str):
        """Load markdown content directly"""
        self.current_content = content
        self.current_file = None
    
    def reload(self):
        """Reload the current file"""
        if self.current_file:
            self.load_markdown_file(self.current_file)
    
    def open_in_browser(self):
        """Open in default browser"""
        if not self.server:
            port = self.start_server()
            print(f"Server started on http://localhost:{port}")
        
        url = f"http://localhost:{self.port}/"
        webbrowser.open(url)
        
        print(f"\n✓ Opened: {os.path.basename(self.current_file) if self.current_file else 'Content'}")
        print(f"✓ URL: {url}")
        print("\nKeep terminal open. Press Ctrl+C to stop.\n")
    
    def run(self):
        """Run the browser and keep it open"""
        self.open_in_browser()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.stop_server()


# Example custom plugin
def create_formula_index_plugin():
    """Example plugin that creates an index of all formulas"""
    return Plugin(
        name="formula-index",
        html_content="""
<div class="formula-index">
    <h3>Formula Index</h3>
    <div id="formula-list"></div>
</div>
""",
        css="""
.formula-index {
    position: fixed;
    right: 20px;
    top: 80px;
    width: 300px;
    max-height: 70vh;
    overflow-y: auto;
    background: white;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 15px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.formula-index h3 {
    margin-top: 0;
    font-size: 16px;
}
#formula-list {
    font-size: 14px;
}
.formula-item {
    margin: 10px 0;
    padding: 5px;
    border-left: 3px solid #0366d6;
    padding-left: 10px;
    cursor: pointer;
}
.formula-item:hover {
    background: #f6f8fa;
}
@media (max-width: 1500px) {
    .formula-index {
        display: none;
    }
}
""",
        javascript="""
// Index all formulas in the document
document.addEventListener('DOMContentLoaded', function() {
    const formulaList = document.getElementById('formula-list');
    const formulas = document.querySelectorAll('.MathJax');
    
    formulas.forEach((formula, index) => {
        const item = document.createElement('div');
        item.className = 'formula-item';
        item.innerHTML = `Formula ${index + 1}: <small>${formula.textContent.substring(0, 30)}...</small>`;
        item.onclick = () => {
            formula.scrollIntoView({ behavior: 'smooth', block: 'center' });
            formula.style.backgroundColor = '#ffffcc';
            setTimeout(() => {
                formula.style.backgroundColor = '';
            }, 2000);
        };
        formulaList.appendChild(item);
    });
});
"""
    )


def main():
    """Main entry point - can be run directly or with a file argument"""
    browser = DirectMarkdownBrowser()
    
    # Example: Register a custom plugin
    # browser.register_plugin(create_formula_index_plugin())
    
    # Determine which file to load
    if len(sys.argv) > 1:
        # File provided as argument
        file_path = sys.argv[1]
    else:
        # Look for default file in current directory
        default_files = ['SamplePaper.md']
        file_path = None
        
        for default_file in default_files:
            if os.path.exists(default_file):
                file_path = default_file
                print(f"Loading default file: {default_file}")
                break
        
        if not file_path:
            # No default file found, show demo content
            print("No markdown file specified and SamplePaper.md not found.")
            print("Usage: python MarkdownBrowser.py [file.md]")
            print("\nShowing demo content...\n")
            
            demo_content = """# MarkdownBrowser Demo

This viewer opens **directly in your browser** with full LaTeX support!

## Features

✓ **No GUI window** - Opens directly in browser  
✓ **Full LaTeX rendering** - Powered by MathJax  
✓ **Plugin system** - Extensible with custom widgets  
✓ **Local server** - No file:// protocol issues  

## Mathematics Examples

Inline math: $E = mc^2$ and $\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$

Display math:
$\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$

## Plugin System

The plugin system allows you to add:
- Custom widgets (TOC, formula index, etc.)
- API endpoints for interactivity
- Custom CSS and JavaScript
- Markdown preprocessors/postprocessors

## Usage

```bash
# With specific file
python MarkdownBrowser.py document.md

# Without arguments (loads SamplePaper.md if found)
python MarkdownBrowser.py
```

Keep the terminal open while viewing. Press Ctrl+C to stop.
"""
            browser.load_markdown_content(demo_content)
            browser.run()
            return
    
    # Load the file
    if browser.load_markdown_file(file_path):
        browser.run()
    else:
        sys.exit(1)


def create_browser(file_path=None, plugins=None):
    """
    Create a browser instance.
    
    Args:
        file_path: Markdown file to load (optional)
        plugins: List of Plugin objects (optional)
    
    Returns:
        DirectMarkdownBrowser instance
    
    Example:
        from MarkdownBrowser import create_browser, Plugin
        
        my_plugin = Plugin(name="my-plugin", html_content="<div>Widget</div>")
        browser = create_browser("doc.md", plugins=[my_plugin])
        browser.run()
    """
    browser = DirectMarkdownBrowser()
    
    if plugins:
        for plugin in plugins:
            browser.register_plugin(plugin)
    
    if file_path:
        browser.load_markdown_file(file_path)
    
    return browser


if __name__ == "__main__":
    main()