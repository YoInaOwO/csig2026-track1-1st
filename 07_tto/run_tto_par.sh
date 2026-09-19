#!/bin/bash
# 在单卡上并行 N 个 TTO 进程处理 <in_dir>，输出 <out_dir>，随后 eval2 打分。
# 用法: bash run_tto_par.sh <in_dir> <out_dir> <tag> <val_dir> <N> [tto args...]
cd /workspace; source venv_odtsr/bin/activate
IN=$1; OUT=$2; TAG=$3; VAL=$4; N=$5; shift 5
mkdir -p $OUT
files=($(ls $IN/*.png | xargs -n1 basename | sed 's/\.png$//'))
for ((k=0; k<N; k++)); do
  sub=""
  for ((j=k; j<${#files[@]}; j+=N)); do sub="$sub,${files[j]}"; done
  sub=${sub#,}
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python tto2.py $IN $OUT --only "$sub" "$@" > logs/tto_${TAG}_$k.log 2>&1 &
done
wait
grep -h -E "^case|TTO_DONE" logs/tto_${TAG}_*.log | grep -v TTO_DONE | sort -V | cut -c1-160
CUDA_VISIBLE_DEVICES=0 python eval2.py $OUT --tag $TAG --val_dir $VAL 2>&1 | grep MEAN | cut -c1-260
echo RUN_TTO_PAR_DONE
