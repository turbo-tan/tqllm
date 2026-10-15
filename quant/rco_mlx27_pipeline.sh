#!/bin/bash
# 27B: uniform GPTQ 4-bit database (RCO code, one bitwidth) -> MLX affine export (embed 8 / head 6). Waits for the GPU.
set -u
O=/mnt/data2t/rco27; mkdir -p $O; L=$O/log.txt; PY=~/.unsloth-venv/bin/python; MPY=~/mlxcpu/bin/python
MODEL=/mnt/data2t/ref/bf16/qwen38-27b-bf16; DB=$O/db; Q=~/code/tanllm-quant
log() { echo "$(date +%T) $*" | tee -a $L; }
until ! pgrep -x "glm-gpu|tqserve|tqkld|llama-perplexity|tqrun" >/dev/null && ! tmux has-session -t yarn 2>/dev/null; do sleep 60; done
log "=== 27B db start"
cd ~/code/RCO
$PY -m torch.distributed.run --nproc-per-node=1 run_quantize.py --model_name_or_path $MODEL \
  --quantizable_modules ".*(q|k|v|o|gate|up|down)_proj$|.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$" \
  --pre_block_modules model.embed_tokens --block_modules model.layers --post_block_modules model.norm lm_head \
  --calibration_data fineweb_edu --calibration_tokens 2097152 --calibration_sequence_length 2048 \
  --bitwidth_options 4 --calibration_bitwidth 4 --group_size 128 --perchannel --dtype bfloat16 --save_dir $DB > $O/db.log 2>&1 < /dev/null
log "db rc=$? $(find $DB -maxdepth 3 -name \"*_qparams.pt\" | wc -l) qparams"
DBL=$(ls -d $DB/*/4bit | head -1)
$MPY $Q/rco_to_mlx.py --model $MODEL --uniform-bw 4 --embed-bits 8 --head-bits 6 --layer-dir $DBL --out $O/mlx_u4eh 2>&1 | tee -a $L
du -sh $O/mlx_u4eh | tee -a $L
log "EXPORT_DONE"
