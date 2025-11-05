categories=(
  "largetable_carry"
  "largetable_lift"
  "smallbox"
  "clothesstand_left_hand"
  "clothesstand_right_hand"
)
tasks=(
  "a person is running straight"
  "a person is running backwards"
  "a person is jumping forward"
)
for i in "${!tasks[@]}"; do
  for j in "${!categories[@]}"; do
    python src/david/joint2smplx.py \
      --dataset FullBodyManip \
      --category ${categories[j]} \
      --epoch 0 \
      --task "${tasks[i]}" \
      --human_motion_dir results/inference_251105/human_motion \
      --device 1
    echo "[Done] task '${tasks[i]}', category '${categories[j]}'"
  done
done