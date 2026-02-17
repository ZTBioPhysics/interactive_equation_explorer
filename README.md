# Interactive Equation Explorer

> **This project has been merged into [Equation_Annotator](https://github.com/berndsen-lab/Equation_Annotator).**
>
> All files (`generate_explorer.py`, `templates/explorer_template.html`, `examples/`) are now in the Equation_Annotator repository. The two tools share the same JSON spec format and conda environment (`equation_annotator`).
>
> **Please use the Equation_Annotator repo going forward.**

---

## Migration

If you were using this repo, switch to:

```bash
cd /path/to/Equation_Annotator

# Standalone explorer (same CLI):
python generate_explorer.py -i examples/michaelis_menten.json --open
python generate_explorer.py --batch-dir examples/

# Or via auto_annotate.py with --explorer flag:
python auto_annotate.py --spec-file output/the_hill_equation.json --explorer
```

The conda environment has been renamed to `equation_annotator`:

```bash
conda activate equation_annotator
```

---

*Original description below for reference.*

Generate interactive HTML pages from annotated equation specs. Each page features a color-coded equation with labels, connector lines, and group brackets — plus optional sliders and a live-updating plot for exploring how parameters affect the equation.
