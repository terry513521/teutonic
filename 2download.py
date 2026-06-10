from hippius_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="mastertensor/teutonic-q3-10b-5ek5koe5-700225159211-rn",
    revision="5Ek5KoE5-700225159211-rn",
    allow_patterns=["*.safetensors", "*.json", "*.py"],
    ignore_patterns="optimizer*",
    max_workers=1,
)

print(local_dir)