categories=(
  "largetable_carry"
  "largetable_lift"
  "smallbox"
  "clothesstand_left_hand"
  "clothesstand_right_hand"
  "clothesstand_two_hands"
)
for i in "${!categories[@]}"; do
  nohup python -u src/david/train_lora.py \
    --david_dataset FullBodyManip \
    --category ${categories[i]} \
    --checkpoint_save_dir "results/david_retarget/lora" \
    --num_steps 10000 \
    --device 0 \
    --seed 20 \
    --overwrite \
    > logs/train_lora_${categories[i]}.out 2>&1 &
done