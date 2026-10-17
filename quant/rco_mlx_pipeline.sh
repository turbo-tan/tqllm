#!/bin/bash
# GB10 chain: RCO per-channel GPTQ database for Qwen3.5-9B -> MLX affine exports -> standard KLD (llama.cpp scorer).
# Arms: our GPTQ uniform 4-bit g128, our GPTQ+RCO 3.5-bit mixed, stock RTN 4-bit g64 (mlx_lm default) and RTN 4-bit g128.
set -u
O=~/rco/mlx; mkdir -p $O; L=$O/log.txt; PY=~/.unsloth-venv/bin/python; MPY=~/mlxcpu/bin/python
MODEL=/home/awee/models/Qwen3.5-9B-text; DB=/home/awee/rco/q35_9b_db_pc; DBL=$DB/Qwen3.5-9B-text/4bit
Q=~/code/tanllm-quant; PPL=~/code/llama.cpp-tq3/build-gx10/bin/llama-perplexity; CONV=~/code/llama.cpp-tq3/convert_hf_to_gguf.py
CFG=~/rco/fix/projected_gumbel_kl_bw2-3-4_target3.5_fineweb_edu_32s_grpnogrp.txt
log() { echo "$(date +%T) $*" | tee -a $L; }
until ! tmux has-session -t yarn 2>/dev/null && ! pgrep -x "tqserve|tqkld|llama-perplexity|tqrun" >/dev/null; do sleep 60; done
log "=== rco->mlx start"
if [ ! -d $DBL ] || [ "$(ls -d $DBL/model.* 2>/dev/null | wc -l)" -lt 200 ]; then
  cd ~/code/RCO
  $PY -m torch.distributed.run --nproc-per-node=1 run_quantize.py --model_name_or_path $MODEL \
    --quantizable_modules ".*(q|k|v|o|gate|up|down)_proj$|.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$" \
    --pre_block_modules model.embed_tokens --block_modules model.layers --post_block_modules model.norm lm_head \
    --calibration_data fineweb_edu --calibration_tokens 2097152 --calibration_sequence_length 2048 \
    --bitwidth_options 2 3 4 --calibration_bitwidth 4 --group_size 128 --perchannel --dtype bfloat16 --save_dir $DB > $O/db.log 2>&1 < /dev/null
  log "db rc=$? entries $(ls -d $DBL/model.* 2>/dev/null | wc -l)"
fi
score() {  # name hf_dir
  local G=$O/$1.gguf
  $PY $CONV $2 --outtype bf16 --outfile $G > $O/conv_$1.log 2>&1 || { log "$1 CONVERT_FAILED"; return; }
  $PPL -m $G --kl-divergence-base ~/models/kldstd/base9b.kld --kl-divergence -c 512 -b 512 -ngl 99 > $O/kld_$1.log 2>&1
  log "$1 $(grep -E 'Mean    KLD' $O/kld_$1.log | cut -c1-60) | $(grep -E 'Same top p' $O/kld_$1.log | cut -c1-60)"
  rm -f $G
}
for name in u4 rco35; do
  case $name in rco35) A="--config $CFG";; u4) A="--uniform-bw 4";; esac
  $MPY $Q/rco_to_mlx.py --model $MODEL $A --layer-dir $DBL --out $O/mlx_$name 2>&1 | tee -a $L
  du -sh $O/mlx_$name | tee -a $L
  $MPY $Q/mlx_dequant_hf.py --mlx $O/mlx_$name --out $O/hf_$name 2>&1 | tail -1 | tee -a $L
  $PY -c "import json,sys; p=sys.argv[1]+'/config.json'; c=json.load(open(p)); c['mtp_num_hidden_layers']=0; json.dump(c,open(p,'w'),indent=1)" $O/hf_$name
  score $name $O/hf_$name; rm -rf $O/hf_$name
done
for g in 64 128; do
  $MPY $Q/mlx_dequant_hf.py --rtn-bits 4 --rtn-group $g --model $MODEL --out $O/hf_rtn4g$g 2>&1 | tail -1 | tee -a $L
  $PY -c "import json,sys; p=sys.argv[1]+'/config.json'; c=json.load(open(p)); c['mtp_num_hidden_layers']=0; json.dump(c,open(p,'w'),indent=1)" $O/hf_rtn4g$g
  score rtn4g$g $O/hf_rtn4g$g; rm -rf $O/hf_rtn4g$g
done
log "RCO_MLX_DONE (earlier fake-quant runs: u4 0.0323, rco35 0.0933, type-rule 0.0953)"
