categories=(
  "largetable_carry"
  "largetable_lift"
  "smallbox"
  "clothesstand_left_hand"
  "clothesstand_right_hand"
  "clothesstand_two_hands"
)
export CUDA_VISIBLE_DEVICES=1
for i in "${!categories[@]}"; do
  nohup python -u src/david/train_omdm.py \
    --dataset FullBodyManip \
    --category ${categories[i]} \
    --hoi_data_dir results/david_retarget/hoi_data \
    --omdm_dir results/david_retarget/omdm \
    --n_epochs 100000 \
    > logs/train_omdm_${categories[i]}.out 2>&1 &
done