categories=(
  "clothesstand_left_hand"
  "clothesstand_right_hand"
  "clothesstand_two_hand"
  "largetable_two_hand_carry"
  "largetable_two_hand_drag"
  "largetable_two_hand_lift"
  "smallbox_two_hand_carry"
  "smallbox_two_hand_drag"
)
tasks=(
  "A person runs fast forward"
  "A person jumps forward"
  "A person high kicks"
  "A person side steps fast"
)
export CUDA_VISIBLE_DEVICES=1
for epoch in 1000 2000 4000 6000 8000; do
  for i in "${!tasks[@]}"; do
    for j in "${!categories[@]}"; do
      prompt="${tasks[i]}, ${categories[j]}"
      pids=()
      for seed in $(seq 0 9); do
        nohup python src/david/inference_mdm.py \
          --david_dataset FullBodyManip \
          --category ${categories[j]} \
          --text_prompt "$prompt" \
          --seed $seed \
          --num_samples 1 \
          --num_repetitions 1 \
          --lora_weight 0.9 \
          --inference_epoch $epoch \
          > logs/inference_mdm_${seed}.out 2>&1 &
        pids+=($!)
      done
      for pid in "${pids[@]}"; do
        wait $pid
      done
    done
  done
done