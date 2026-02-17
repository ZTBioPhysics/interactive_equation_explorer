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
# Can also render specs from the Equation_Annotator project directly:
python generate_explorer.py -i /path/to/Equation_Annotator/output/the_hill_equation.json --open
```

## Current Status
- Full compatibility with Equation_Annotator JSON spec format
- Supports two input modes:
  - **`interactive` block** (explorer-native): single math.js expression with named variables/sliders
  - **`plot` block** (from Equation_Annotator): multi-curve numpy expressions, auto-converted to math.js, with annotations and auto-generated sliders
- Supported spec fields: `title`, `segments` (with `superscript`), `groups`, `description`, `symbols` (grouped by type), `constants` (legacy), `use_cases`, `insight`, `interactive`, `plot`
- Label overlap detection spreads crowded annotations automatically

## Key Dependencies
- Python: jinja2
- CDN: KaTeX, Plotly.js, math.js (loaded in HTML)
- Conda env: `interactive_equation_explorer`

## Companion Project
- **Equation_Annotator** — generates static annotated equation images (matplotlib/HTML). Located at `../Equation_Annotator/`. Shares the same JSON spec format; output JSON files can be fed directly into this explorer.

## Potential Next Steps
- Add more example specs with `interactive` blocks
- Support `display_mode` filtering (full/compact/plot/insight/minimal) from annotator specs
- Consider merging `interactive` and `plot` block handling into a unified path
