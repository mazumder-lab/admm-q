torchrun --nnodes=1 --nproc_per_node=1 --master_port=$((RANDOM % 10000 + 20000)) optimize_rotation.py \
    --input_model $MODEL_PATH \
    --output_rotation_path "${ROT_DIR}/" \
    --output_dir "${WORK_DIR}/replication/llama3_1B_instruct/results_w4a4kv4/" \
    --logging_dir "${WORK_DIR}/replication/llama3_1B_instruct/logs/tb_w4a4kv4/" \
    --model_max_length 2048 \
    --fp16 False \
    --bf16 True \
    --log_on_each_node False \
    --per_device_train_batch_size 8 \
    --logging_steps 1 \
    --learning_rate 1.5 \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --save_safetensors False \
    --max_steps 100 \
    --w_bits 4 \
    --a_bits 4 \
    --k_bits 4 \
    --v_bits 4 \
    --w_clip \
    --a_asym \
    --k_asym \
    --v_asym \
    --k_groupsize 64 \
    --v_groupsize 64

EXIT_CODE=$?
echo "================================================================================"
echo "Rotation optimization completed with exit code: $EXIT_CODE | End: $(date)"
echo "================================================================================"
