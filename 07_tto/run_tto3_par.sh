#!/bin/bash
# 单卡并行 N 个 tto3 进程处理 <in_dir> -> <out_dir>，随后 eval2 打分
# 用法: bash run_tto3_par.sh <in_dir> <out_dir> <tag> <val_dir> <N> [tto3 args...]
cd /workspace; source venv_odtsr/bin/activate
IN=$1; OUT=$2; TAG=$3; VAL=$4; N=$5; shift 5
mkdir -p $OUT
files=($(ls $IN/*.png $IN/*.jpg 2>/dev/null | xargs -n1 basename | sed 's/\.[a-z]*$//'))
for ((k=0; k<N; k++)); do
  sub=""
  for ((j=k; j<${#files[@]}; j+=N)); do sub="$sub,${files[j]}"; done
  sub=${sub#,}
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python tto3.py $IN $OUT --only "$sub" "$@" > logs/tto_${TAG}_$k.log 2>&1 &
  sleep 3
done
wait
grep -h -E "^case" logs/tto_${TAG}_*.log | sort -V | cut -c1-200
grep -h -E "Error|Traceback" logs/tto_${TAG}_*.log | head -3
CUDA_VISIBLE_DEVICES=0 python eval2.py $OUT --tag $TAG --val_dir $VAL 2>&1 | grep -A1 MEAN | cut -c1-260
echo RUN_TTO_PAR_DONE
