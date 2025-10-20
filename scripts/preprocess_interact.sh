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
sudo mkdir logs
for i in "${!categories[@]}"; do
  nohup python -u src/david/preprocess_interact.py \
    --input_dir data/InterAct \
    --pose_dir results/david/pose_data \
    --dataset FullBodyManip \
    --category ${categories[i]} \
    > logs/preprocess_${categories[i]}.out 2>&1 &
done