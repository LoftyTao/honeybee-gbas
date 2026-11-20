[中文](README.md) | English

# honeybee-gbas

<p>
  <img src="assets/honeybee-logo.png" alt="Honeybee" height="72" />
  <img src="assets/gbas-logo.png" alt="GBAS2026" height="72" />
</p>

`honeybee-gbas` is a **Honeybee** extension for exporting **Honeybee** models to native **GBAS2026** DWG files.
The distribution package name is `honeybee-gbas`, and the Python import name is `honeybee_gbas`.
The current release version is `1.0.0`.
`model_to_dwg` serves as the single public export entry point and writes native binary DWG drawings directly in pure Python.
Export coverage includes model geometry, room boundaries, window and door openings, roof skylights, shades, and source thermal properties.

## Sydney Opera House Showcase

<p>
  <img src="assets/sydney-opera-house.png" alt="Sydney Opera House" width="680" />
</p>

The showcase above presents the Sydney Opera House model across building simulation formats, with the bottom branch showing the native DWG exported by `honeybee-gbas`.

## Requirements and Prerequisites

The runtime environment requires Python 3.9 or higher.
Package dependencies include `honeybee-core`, `honeybee-energy`, and `honeybee-energy-standards`.
Geometry primitives and spatial operations rely on `ladybug-geometry` and `shapely` 2.x.
Installing with `pip` resolves and installs all Python dependencies automatically.
Project configuration, weather settings, and building performance simulation run inside the **GBAS2026** runtime environment.

## Installation

Install from source:

```console
python -m pip install .
```

Verify the command-line entry points:

```console
honeybee-gbas --help
honeybee-gbas translate --help
python -m honeybee_gbas translate model-to-dwg --help
```

## Quick Start

```python
from honeybee.model import Model
from honeybee_gbas.writer import model_to_dwg

model = Model.from_file("model.hbjson")
dwg_path, report = model_to_dwg(model, "model.dwg")
print(dwg_path)
```

Public API signature:

```text
model_to_dwg(model, output_file, report_path=None, overwrite=False) -> tuple[Path, dict]
```

`model_to_dwg` accepts a **Honeybee** `Model` object and is also registered on `model.to.dwg(model, "model.dwg")`.
Default export writes the DWG file and returns `(Path, dict)` containing the written path and self-check report.
Specify `report_path` to serialize the export report as JSON:

```python
model_to_dwg(
    model,
    "model.dwg",
    report_path="model.report.json",
    overwrite=True,
)
```

Set `overwrite=True` to replace existing destination files.
Before exporting, `model.check_for_extension("gbas")` and `model.properties.gbas` validate room presence and horizontal floor boundaries.

## Command Line Interface

```console
honeybee-gbas translate model-to-dwg model.hbjson -o model.dwg
```

By default, the CLI writes to the source filename stem with a `.dwg` extension.
Pass `--report` and `--overwrite` to write a JSON report and allow replacing existing files:

```console
honeybee-gbas translate model-to-dwg model.hbjson -o model.dwg --report model.report.json --overwrite
```

Running via Python module is equivalent:

```console
python -m honeybee_gbas translate model-to-dwg model.hbjson -o model.dwg
```

## Sample Models

The `samples` directory in the repository root organizes all 12 test models into per-case subdirectories.
Each subdirectory contains `<case>.hbjson`, the exported `<case>.dwg` produced by `honeybee-gbas`, and a standalone `<case>.html` 3D model preview.

## Uninstallation

```console
python -m pip uninstall honeybee-gbas
```

## License

This project is licensed under AGPL-3.0-only.
Derived DWG binary backend modules from ezdwg retain the MIT license.
Complete license texts are stored in `LICENSE` and `honeybee_gbas/dwg/LICENSE-ezdwg.txt`.
