torchrun --nnodes=1 --nproc_per_node=1 --master_port=$((RANDOM % 10000 + 20000)) ptq.py \
    --input_model $MODEL_PATH \
    --do_train False \
    --do_eval True \
    --per_device_eval_batch_size 4 \
    --model_max_length 2048 \
    --fp16 False \
    --bf16 True \
    --save_safetensors False \
    --w_bits 4 \
    --a_bits 4 \
    --k_bits 4 \
    --v_bits 4 \
    --w_clip \
    --w_admm \
    --a_asym \
    --k_asym \
    --v_asym \
    --k_groupsize 64 \
    --v_groupsize 64 \
    --rotate \
    --optimized_rotation_path "${ROT_PATH}" \
    --eval_zeroshot

EXIT_CODE=$?
echo "================================================================================"
echo "Eval completed with exit code: $EXIT_CODE | End: $(date)"
echo "================================================================================"
