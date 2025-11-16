# x-ray

**Requirements**

- **Python**: `>=3.11`.
- **Packages**: this project lists dependencies in `pyproject.toml` (numpy, pillow, streamlit, torch, torchvision).

**Quick Install**

- **Create virtual env**: `python -m venv .venv`.
- **Activate (bash)**: `source .venv/Scripts/activate` (Windows Git Bash) or `source .venv/bin/activate` (Linux/macOS).
- **Activate (PowerShell)**: `.venv\Scripts\Activate.ps1`.
- **Upgrade pip**: `python -m pip install --upgrade pip`.
- **Install dependencies from project**: run `pip install .` (this will install packages from `pyproject.toml`).

If you prefer to install packages directly, you can run:

```
python -m pip install numpy pillow streamlit torch torchvision
```

**Model file**

- The repository includes `model.pth` in the project root. Keep the model file next to `app.py` (or update paths in the code if you move it).
- If `model.pth` is not present, place the downloaded model file at the repository root with the filename `model.pth`.

**Run the app**

- **Streamlit app**: `streamlit run app.py`.
- **Direct run (if available)**: `python main.py`.

**Notes & Troubleshooting**

- **CUDA / GPU**: install a PyTorch build that matches your CUDA version if you need GPU support. Use the official PyTorch installation instructions for CUDA-enabled builds.
- **Windows PowerShell policy**: if activation fails in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then activate.
- **Missing dependency errors**: activate your virtual environment and re-run `pip install .`.

**Next steps**

- To run locally: create and activate a venv, install dependencies, then `streamlit run app.py`.
- Tell me if you want a `requirements.txt`, a Dockerfile, or a one-shot install script; I can add it.
