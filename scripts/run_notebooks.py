"""Execute all notebooks in place. Используется из run_all.bat."""
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


NOTEBOOKS = [
    'notebooks/01_eda.ipynb',
    'notebooks/02_preprocessing.ipynb',
    'notebooks/03_baseline.ipynb',
    'notebooks/04_models.ipynb',
    'notebooks/05_error_analysis.ipynb',
]


def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    for path in NOTEBOOKS:
        print(f'--- {path} ---', flush=True)
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(
            nb,
            timeout=900,
            kernel_name='python3',
            resources={'metadata': {'path': str(Path(path).parent)}},
        )
        client.execute()
        nbformat.write(nb, path)
        print(f'ok: {path}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
