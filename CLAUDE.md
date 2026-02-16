# Interactive Equation Explorer

Generates interactive HTML files from annotation JSON specs. Users can adjust equation parameters via sliders and see live-updating plots. Complements the static Equation_Annotator project.

## Project Structure
- `generate_explorer.py` — Python generator (reads JSON, writes HTML via Jinja2)
- `templates/explorer_template.html` — HTML/CSS/JS template
- `examples/` — Example JSON specs (annotation + optional interactive block)
- `output/` — Generated HTML files (gitignored)

## Usage
```bash
python generate_explorer.py -i examples/michaelis_menten.json --open
python generate_explorer.py --batch-dir examples/
```

## Current Status
- Initial implementation

## Key Dependencies
- Python: jinja2
- CDN: KaTeX, Plotly.js, math.js (loaded in HTML)
