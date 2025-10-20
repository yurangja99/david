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
export CUDA_VISIBLE_DEVICES=0
for epoch in 1000 2000 4000 6000 8000; do
  for i in "${!categories[@]}"; do
    nohup python -u src/david/joint2smplx.py \
      --dataset FullBodyManip \
      --category ${categories[i]} \
      --epoch ${epoch} \
      --task "${tasks[i]}" \
    > logs/joint2smplx_${categories[i]}.out 2>&1 &
  done
done