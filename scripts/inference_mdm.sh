categories=(
  "largetable_carry"
  "largetable_lift"
  "smallbox"
  "clothesstand_left_hand"
  "clothesstand_right_hand"
)
# objects=(
#   "a table"
#   "a table"
#   "a small box"
#   "a long object"
#   "a long object"
# )
ref_idx=(0 0 0 0 0)
blending_frame=(10 10 10 10 10)
smoothing_frame=(5 5 5 5 5 5)
hand_info=(0 0 0 1 2)
tasks=(
  "a person is running straight"
  "a person is running backwards"
  "a person is jumping forward"
  "a person is doing a high kick"
  "a person is dancing an energetic cha-cha"
)
for i in "${!tasks[@]}"; do
  for j in "${!categories[@]}"; do
    prompt="${tasks[i]}" # while carrying an object"
    pids=()
    for seed in $(seq 0 9); do
      nohup python -u src/david/inference_mdm.py \
        --david_dataset FullBodyManip \
        --category ${categories[j]} \
        --lora_dir results/david_retarget_251028/lora \
        --human_motion_dir results/inference_251105/human_motion \
        --text_prompt "$prompt" \
        --seed $seed \
        --num_samples 1 \
        --num_repetitions 1 \
        --lora_weight 0.0 \
        --inference_epoch 0 \
        --ref_dir results/david_retarget_251028/mdm \
        --ref_idx ${ref_idx[j]} \
        --ref_bf ${blending_frame[j]} \
        --ref_sf ${smoothing_frame[j]} \
        --hand_info ${hand_info[j]} \
        --device 0 \
        >> logs/inference_mdm_${seed}.out 2>&1 &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do
      wait $pid
    done
    echo "[Done] task '${tasks[i]}', category '${categories[j]}'"
  done
done