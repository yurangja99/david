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
for i in "${!categories[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python src/david/process_omdm.py \
    --dataset FullBodyManip \
    --category ${categories[i]}
done