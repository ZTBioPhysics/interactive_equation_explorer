# Interactive Equation Explorer

Generate interactive HTML pages from annotated equation specs. Each page features a color-coded equation with labels, connector lines, and group brackets — plus optional sliders and a live-updating plot for exploring how parameters affect the equation.

Complements the [Equation Annotator](https://github.com/berndsen-lab/Equation_Annotator) project, which produces static PNG/SVG images.

## Quick Start

```bash
# Create and activate the conda environment
conda create -n interactive_equation_explorer python=3.11 -y
conda activate interactive_equation_explorer
pip install -r requirements.txt

# Generate a single interactive explorer
python generate_explorer.py -i examples/michaelis_menten.json --open

# Generate all examples at once
python generate_explorer.py --batch-dir examples/
```

Generated HTML files are written to `output/` (gitignored).

## Features

- **Color-coded equations** rendered with KaTeX, matching the Equation Annotator visual style
- **Connector lines** from equation segments to their descriptive labels
- **Group brackets** for hierarchical annotation (multiple nesting levels)
- **Description, constants, and use-case** panels
- **Interactive sliders** for adjusting equation parameters in real time
- **Live Plotly plot** that updates instantly as you drag sliders
- **Annotation-only mode** — specs without an `interactive` block produce a static annotated page (no sliders or plot)
- **Dark theme** (#1a1a2e background) consistent with Equation Annotator output
- **Responsive layout** that works on desktop and mobile

## JSON Spec Format

Each spec is a JSON file with annotation fields (required) and an optional `interactive` block:

```json
{
  "title": "The Michaelis–Menten Equation",
  "segments": [
    { "latex": "$v$", "color": "#FF6B6B", "label": "reaction\nvelocity" },
    { "latex": "$ = $", "color": "#AAAAAA", "label": null },
    { "latex": "$V_{max}$", "color": "#4ECDC4", "label": "maximum\nreaction rate" }
  ],
  "groups": [
    { "segment_indices": [2, 3, 4], "label": "catalytic output", "color": "#4ECDC4", "level": 1 }
  ],
  "description": "Plain-English explanation of the equation.",
  "use_cases": ["Field: Application description"],
  "constants": [{ "symbol": "e", "description": "Euler's number" }],

  "interactive": {
    "expression": "Vmax * S / (Km + S)",
    "output": { "symbol": "v", "label": "Reaction velocity", "latex": "v" },
    "variables": [
      { "name": "S", "label": "Substrate [S]", "latex": "[S]",
        "role": "independent", "min": 0, "max": 100, "default": 50, "step": 0.5, "unit": "mM" },
      { "name": "Vmax", "label": "Max velocity", "latex": "V_{max}",
        "role": "parameter", "min": 0.1, "max": 200, "default": 100, "step": 1, "unit": "mM/s" }
    ],
    "plot": {
      "type": "2d",
      "x_axis": { "variable": "S", "label": "[S] (mM)" },
      "y_axis": { "label": "v (mM/s)" }
    }
  }
}
```

### Key Fields

| Field | Required | Description |
|-------|----------|-------------|
| `title` | Yes | Displayed at the top of the page |
| `segments` | Yes | Array of equation parts with `latex`, `color`, and optional `label` |
| `groups` | No | Bracket annotations spanning multiple segments |
| `description` | No | Plain-English explanation |
| `use_cases` | No | Bulleted list of practical applications |
| `constants` | No | Mathematical constants with descriptions |
| `interactive` | No | Enables sliders and plot (see below) |

### Interactive Block

| Field | Description |
|-------|-------------|
| `expression` | Math.js-compatible expression (e.g., `"Vmax * S / (Km + S)"`) |
| `output` | The dependent variable (`symbol`, `label`, `latex`) |
| `variables` | Array of variables with `role: "independent"` or `role: "parameter"` |
| `plot.type` | Plot type: `"2d"` (line plot) |
| `plot.x_axis` / `plot.y_axis` | Axis labels |

Variables with `role: "parameter"` get sliders. The single `role: "independent"` variable is the x-axis.

## CLI Options

```
python generate_explorer.py [OPTIONS]

  -i, --input FILE        Single JSON spec file
  --batch-dir DIR         Directory of JSON specs (renders all .json files)
  -o, --output DIR        Output directory (default: output/)
  --template-dir DIR      Template directory (default: templates/)
  --open                  Open generated HTML in default browser
```

## Examples

| Example | Type | Description |
|---------|------|-------------|
| `michaelis_menten.json` | Interactive | Enzyme kinetics with Vmax and Km sliders |
| `logistic_growth.json` | Interactive | Population growth with K, r, and A sliders |
| `euler_identity.json` | Annotation-only | Five fundamental constants, no sliders |

## Dependencies

- **Python**: jinja2 (template rendering)
- **CDN** (loaded in generated HTML): KaTeX, Plotly.js, math.js
