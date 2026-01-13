# Maintaining Documentation

This guide explains how to update and maintain the Foresight Metrics documentation.

## Overview

The documentation uses [MkDocs](https://www.mkdocs.org/) with:

- **Material theme** — Modern, responsive design
- **mkdocstrings** — Auto-generates API docs from Python docstrings

## Quick Start

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Serve locally with auto-reload
mkdocs serve

# Open http://127.0.0.1:8000 in your browser
```

## Documentation Structure

```
docs/
├── index.md                    # Homepage
├── getting-started/
│   ├── installation.md         # Installation guide
│   └── quickstart.md           # Quick start tutorial
├── user-guide/
│   ├── object-detection.md     # OD user guide
│   ├── segmentation.md         # Segmentation user guide
│   └── extensions.md           # How to extend
├── api/                        # API Reference (auto-generated)
│   ├── index.md
│   ├── object-detection.md
│   ├── segmentation.md
│   ├── loggers.md
│   └── results.md
└── contributing/
    └── documentation.md        # This file
```

## What Auto-Updates vs Manual Updates

| Content | Auto-updates? | Location |
|---------|---------------|----------|
| Class/method signatures | ✅ Yes | Docstrings in Python code |
| Parameter descriptions | ✅ Yes | Docstrings in Python code |
| Examples in docstrings | ✅ Yes | Docstrings in Python code |
| User guide examples | ❌ No | `docs/user-guide/*.md` |
| Metric/format tables | ❌ No | `docs/user-guide/*.md` |
| Installation steps | ❌ No | `docs/getting-started/installation.md` |

## Updating API Documentation

API docs are generated from Python docstrings using mkdocstrings.

### Example: Updating a Class Description

Edit the docstring in Python:

```python
# foresight_metrics/tasks/object_detection/__init__.py

class ObjectDetection:
    """
    Evaluate object detection models.  # ← This becomes the description
    
    This class handles loading data, computing metrics, and logging.
    
    Attributes:
        data_format: Name of the data format being used.
        
    Example:
        >>> od = ObjectDetection()
        >>> result = od.evaluate("gt.json", "preds.json")
    """
```

The API reference page will automatically reflect these changes on next build.

### Adding a New Class to API Docs

1. Create or edit the relevant `.md` file in `docs/api/`
2. Add the mkdocstrings directive:

```markdown
::: foresight_metrics.path.to.YourClass
    options:
      show_root_heading: true
      show_source: false
```

## Updating User Guides

User guides in `docs/user-guide/` are hand-written. Update them when:

- API signatures change
- New features are added
- Examples become outdated

### Checklist for New Features

When adding a new metric, format, or task:

- [ ] Update "Supported Metrics/Formats" table in relevant user guide
- [ ] Add example usage if needed
- [ ] Update `docs/api/` if new classes are exposed
- [ ] Update `docs/index.md` if it's a major feature

## Writing Good Docstrings

Follow Google-style docstrings for consistency:

```python
def evaluate(self, ground_truth, predictions) -> MetricResult:
    """
    Evaluate predictions against ground truth.
    
    This method loads data, computes metrics, and logs results.
    
    Args:
        ground_truth: Path to GT file or data structure.
        predictions: Path to predictions file or data structure.
        
    Returns:
        MetricResult containing all computed metrics.
        
    Raises:
        ValueError: If data format is invalid.
        FileNotFoundError: If file paths don't exist.
        
    Example:
        >>> od = ObjectDetection()
        >>> result = od.evaluate("gt.json", "preds.json")
        >>> print(result.metrics)
    """
```

## Building and Deploying

### Local Development

```bash
mkdocs serve  # Auto-reloads on changes
```

### Build Static Site

```bash
mkdocs build  # Output in site/
```

### Deploy to GitHub Pages

```bash
mkdocs gh-deploy  # Pushes to gh-pages branch
```

This will make docs available at: https://jf-horton.github.io/fs-cv-metrics/

## Configuration

The MkDocs configuration is in `mkdocs.yml`:

```yaml
# Key settings
site_name: Foresight Metrics
theme:
  name: material

plugins:
  - mkdocstrings  # Auto-generate from docstrings

nav:
  - Home: index.md
  - ...  # Navigation structure
```

## Tips

1. **Keep examples in docstrings** — They auto-update and stay in sync
2. **Use `mkdocs serve`** — See changes instantly while editing
3. **Check links** — Broken links will show warnings during build
4. **Preview before deploying** — Run `mkdocs build` and check `site/`
