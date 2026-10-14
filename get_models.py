from huggingface_hub import snapshot_download
D = "/mnt/data_2t_3/github_repos/azllm/models"
snapshot_download("YTan2000/Qwen3.8-27B-TQ3_4S", local_dir=D, allow_patterns=["Qwen3.8-27B-TQ3_4S-v2.gguf"])
print("model done", flush=True)
snapshot_download("YTan2000/Qwen3.8-27B-TQ", local_dir=D, allow_patterns=["drafters/Qwen3.8-27B-DFlash2-TQ3.gguf","mtp-vocab/atx_65536.txt"])
print("drafter+vocab done", flush=True)
