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
export CUDA_VISIBLE_DEVICES=0
for i in "${!categories[@]}"; do
  nohup python -u src/david/train_lora.py \
    --david_dataset FullBodyManip \
    --category ${categories[i]} \
    --num_steps 10000 \
    > logs/train_lora_${categories[i]}.out 2>&1 &
done