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
for epoch in 2000 10000 100000; do
  for i in "${!categories[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python src/david/inference_omdm.py \
      --dataset FullBodyManip \
      --category ${categories[i]} \
      --inference_contact_threshold 0.05 \
      --inference_epoch $epoch
  done
done