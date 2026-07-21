# Environment setup (Lane A data)

From the **repo root**:

```bash
# 1. Virtualenv + deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download corpora
python scripts/download_data.py --smoke-only   # small slices (smoke)
# or
python scripts/download_data.py                # full TinyStories + Simple English Wikipedia
```

On Windows (PowerShell): `.venv\Scripts\Activate.ps1` instead of `source .venv/bin/activate`.

Outputs land under `data/raw/`; details in [`README.md`](./README.md) and licenses in [`LICENSES.md`](./LICENSES.md).
