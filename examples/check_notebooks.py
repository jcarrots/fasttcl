"""Execute the example notebooks with the Python environment running this script."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import json

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/notebooks"))
    parser.add_argument("--in-place", action="store_true", help="Also save executed outputs into the source notebooks.")
    args = parser.parse_args()
    examples = Path(__file__).resolve().parent
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    notebooks = sorted(examples.glob("*.ipynb"))
    if not notebooks:
        raise SystemExit("No example notebooks found.")

    with tempfile.TemporaryDirectory(prefix="kernel-", dir=output_dir) as temporary:
        runtime = Path(temporary)
        kernel_root = runtime / "kernels"
        spec_dir = kernel_root / "python3"
        spec_dir.mkdir(parents=True)
        (spec_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Example validation",
            "language": "python",
        }), encoding="utf-8")
        environment = os.environ.copy()
        for name, folder in (("MPLCONFIGDIR", "matplotlib"), ("IPYTHONDIR", "ipython"),
                             ("JUPYTER_RUNTIME_DIR", "jupyter")):
            path = runtime / folder
            path.mkdir()
            environment[name] = str(path)

        for source in notebooks:
            notebook = nbformat.read(source, as_version=4)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cell.outputs = []
                    cell.execution_count = None
            nbformat.validate(notebook)
            manager = KernelManager(
                kernel_name="python3",
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]),
                connection_file=str(runtime / "connection.json"),
            )
            print(f"Executing {source.name} ...", flush=True)
            NotebookClient(
                notebook, km=manager, timeout=300,
                resources={"metadata": {"path": str(examples)}},
                record_timing=False,
            ).execute(env=environment, cleanup_kc=True)
            nbformat.validate(notebook)
            code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
            if any(cell.execution_count is None for cell in code_cells):
                raise RuntimeError(f"Unexecuted cell in {source.name}")
            figures = sum(
                "image/png" in output.get("data", {})
                for cell in code_cells for output in cell.outputs
            )
            destination = output_dir / source.name
            nbformat.write(notebook, destination)
            if args.in_place:
                nbformat.write(notebook, source)
            print(f"  Passed: {len(code_cells)} code cells, {figures} figures. Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
