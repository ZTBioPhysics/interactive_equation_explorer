#!/usr/bin/env python3
"""
Interactive Equation Explorer — HTML Generator

Reads annotation JSON specs (with optional interactive block) and produces
self-contained HTML files with:
  - Color-coded KaTeX equation with labels, connectors, and group brackets
  - Slider controls for adjustable parameters
  - Live-updating Plotly plot

Usage:
    python generate_explorer.py -i examples/michaelis_menten.json --open
    python generate_explorer.py --batch-dir examples/
"""

import argparse
import json
import re
import webbrowser
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# ============================================================================
# CONFIGURATION - Edit these for Spyder / interactive use
# ============================================================================
INPUT_FILE = None               # Path to a single JSON spec file
BATCH_DIR = None                # Path to directory of JSON specs (renders all)
OUTPUT_DIR = "output"           # Where to write generated HTML files
TEMPLATE_DIR = "templates"      # Directory containing Jinja2 templates
OPEN_IN_BROWSER = False         # Auto-open generated HTML in default browser
# ============================================================================


def load_spec(path):
    """Load and validate a JSON annotation spec.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.

    Returns
    -------
    dict
        The parsed spec with at least 'title' and 'segments'.

    Raises
    ------
    ValueError
        If required fields are missing.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        data = {"title": path.stem.replace("_", " ").title(), "segments": data}

    if "segments" not in data:
        raise ValueError(f"{path}: JSON must contain a 'segments' key.")

    data.setdefault("title", "Equation Explorer")
    data.setdefault("groups", [])
    data.setdefault("description", None)
    data.setdefault("use_cases", [])
    data.setdefault("constants", [])

    return data


def _clean_latex(raw):
    """Strip surrounding $ signs from a LaTeX string for KaTeX rendering."""
    return raw.strip().lstrip("$").rstrip("$").strip()


def _prepare_segments(segments):
    """Add template-friendly fields to each segment dict."""
    prepared = []
    for seg in segments:
        s = dict(seg)
        s["latex_clean"] = seg["latex"]  # Keep original for data-attr
        s["label_html"] = (seg.get("label") or "").replace("\n", "<br>")
        prepared.append(s)
    return prepared


def _get_group_levels(groups):
    """Return sorted list of unique group levels."""
    if not groups:
        return []
    levels = sorted({g.get("level", 1) for g in groups})
    return levels


def render_html(spec, template_dir, output_path):
    """Render a spec to an HTML file using the Jinja2 template.

    Parameters
    ----------
    spec : dict
        Loaded annotation spec.
    template_dir : str or Path
        Directory containing explorer_template.html.
    output_path : str or Path
        Where to write the generated HTML.

    Returns
    -------
    Path
        The output file path.
    """
    template_dir = Path(template_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,
    )
    template = env.get_template("explorer_template.html")

    has_interactive = "interactive" in spec and spec["interactive"] is not None
    segments = _prepare_segments(spec["segments"])
    groups = spec.get("groups", [])
    # Ensure each group has a level field
    for g in groups:
        g.setdefault("level", 1)

    group_levels = _get_group_levels(groups)

    # Extract parameter variables for slider rendering
    parameters = []
    all_variables = []
    if has_interactive:
        for v in spec["interactive"].get("variables", []):
            if v.get("role") == "parameter":
                parameters.append(v)
            all_variables.append(v)

    # Auto-generate variable descriptions from interactive block when no
    # explicit constants are provided.  This ensures every interactive spec
    # gets a variables section without duplicating info into a constants block.
    constants = spec.get("constants", [])
    auto_variables = []
    if not constants and all_variables:
        output = spec["interactive"].get("output", {})
        # Include the output variable first
        if output:
            unit_str = ""
            # Try to infer unit from y-axis label
            y_label = spec["interactive"].get("plot", {}).get("y_axis", {}).get("label", "")
            if "(" in y_label:
                unit_str = y_label.split("(")[-1].rstrip(")")
            auto_variables.append({
                "symbol": output.get("latex", output.get("symbol", "")),
                "description": output.get("label", ""),
                "unit": unit_str,
            })
        for v in all_variables:
            unit_str = v.get("unit", "")
            desc = v.get("label", v["name"])
            if unit_str:
                desc += f" ({unit_str})"
            role_tag = "independent variable" if v.get("role") == "independent" else "parameter"
            auto_variables.append({
                "symbol": v.get("latex", v["name"]),
                "description": f"{desc} — {role_tag}",
            })

    html = template.render(
        title=spec.get("title", "Equation Explorer"),
        segments=segments,
        groups=groups,
        group_levels=group_levels,
        description=spec.get("description"),
        use_cases=spec.get("use_cases", []),
        constants=constants,
        auto_variables=auto_variables,
        has_interactive=has_interactive,
        parameters=parameters,
        spec_json=json.dumps(spec, indent=None),
    )

    output_path.write_text(html, encoding="utf-8")
    print(f"  Generated: {output_path}")
    return output_path


def _output_name(input_path):
    """Derive an output filename from an input JSON path."""
    stem = Path(input_path).stem
    # Sanitize: keep alphanumeric + underscores
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", stem)
    return f"{safe}.html"


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML equation explorers from JSON specs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_explorer.py -i examples/michaelis_menten.json --open
  python generate_explorer.py --batch-dir examples/
  python generate_explorer.py -i spec.json -o my_output/
        """,
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Path to a single JSON spec file.",
    )
    parser.add_argument(
        "--batch-dir", type=str, default=None,
        help="Directory of JSON specs to render (all .json files).",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--template-dir", type=str, default=None,
        help=f"Template directory (default: {TEMPLATE_DIR}).",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Open generated HTML in default browser.",
    )
    args = parser.parse_args()

    # Resolve settings: CLI args > configuration constants
    input_file = args.input or INPUT_FILE
    batch_dir = args.batch_dir or BATCH_DIR
    output_dir = Path(args.output or OUTPUT_DIR)
    template_dir = Path(args.template_dir or TEMPLATE_DIR)
    open_browser = args.open or OPEN_IN_BROWSER

    if not template_dir.is_dir():
        parser.error(f"Template directory not found: {template_dir}")

    # Collect input files
    input_files = []
    if input_file:
        input_files.append(Path(input_file))
    if batch_dir:
        batch_path = Path(batch_dir)
        if not batch_path.is_dir():
            parser.error(f"Batch directory not found: {batch_path}")
        input_files.extend(sorted(batch_path.glob("*.json")))

    if not input_files:
        parser.error(
            "No input specified. Use --input for a single file "
            "or --batch-dir for a directory of specs."
        )

    # Render each spec
    generated = []
    for input_path in input_files:
        print(f"Processing: {input_path}")
        spec = load_spec(input_path)
        out_name = _output_name(input_path)
        out_path = output_dir / out_name
        render_html(spec, template_dir, out_path)
        generated.append(out_path)

    print(f"\nDone — {len(generated)} file(s) generated in {output_dir}/")

    # Open the first (or only) file in browser
    if open_browser and generated:
        url = generated[0].resolve().as_uri()
        print(f"Opening: {url}")
        webbrowser.open(url)


if __name__ == "__main__":
    main()
