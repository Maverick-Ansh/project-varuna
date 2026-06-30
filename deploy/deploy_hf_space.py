"""Turnkey: create + push the Varuna backend to a Hugging Face Docker Space.

Your HF write token is read from the HF_TOKEN env var — it is never printed or stored. Run:

    HF_TOKEN=hf_xxx python deploy/deploy_hf_space.py [owner/space-name]

Default space id: Maverick-Ansh/varuna-floodtwin. After it finishes, the Space builds the Dockerfile
and serves on port 7860; smoke-test https://<owner>-<name>.hf.space/api/health.

Optional chat: add a free Groq key (https://console.groq.com/keys) as a Space secret LLM_API_KEY.
"""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SPACE = "Maverick-Ansh/varuna-floodtwin"

# Only the backend + bundle go to the Space (keep the image lean).
ALLOW = ["Dockerfile", "requirements-deploy.txt", "varuna/**", "api/**", "artifacts/**"]
IGNORE = [
    "**/__pycache__/**", "*.pyc",
    "artifacts/patna/observed_water_*.tif",   # masks not needed at serve time
    "artifacts/patna/twin_dataset.pt",        # big, training-only
    "web/**", "tests/**", "notebooks/**", ".git/**",
]


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN env var not set. Get a write token at "
                 "https://huggingface.co/settings/tokens and run:\n"
                 "  HF_TOKEN=hf_xxx python deploy/deploy_hf_space.py [owner/space]")
    space_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SPACE

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    api.create_repo(repo_id=space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"Space ready: {space_id}")

    # HF reads the Space card front-matter from README.md
    api.upload_file(path_or_fileobj=os.path.join(REPO_ROOT, "deploy", "space_README.md"),
                    path_in_repo="README.md", repo_id=space_id, repo_type="space")

    api.upload_folder(folder_path=REPO_ROOT, repo_id=space_id, repo_type="space",
                      allow_patterns=ALLOW, ignore_patterns=IGNORE,
                      commit_message="Deploy Varuna FloodTwin backend")

    owner, name = space_id.split("/")
    sub = f"{owner}-{name}".lower().replace("_", "-")
    print(f"\nDeployed. Building now (a few min). Endpoints:\n"
          f"  https://huggingface.co/spaces/{space_id}   (logs/build)\n"
          f"  https://{sub}.hf.space/api/health          (smoke test)\n"
          f"Set VITE_API_BASE=https://{sub}.hf.space on Vercel for the frontend.")


if __name__ == "__main__":
    main()
