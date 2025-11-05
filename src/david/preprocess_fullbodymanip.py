import os, argparse, glob, joblib, numpy as np
from pathlib import Path
import json
import torch
from src.david.process_omdm import plot_3d_points, get_object_vertices
from imports.mdm.visualize import vis_utils

def normalize(s: str) -> str:
    # 대소문자/언더스코어/하이픈/띄어쓰기 차이를 없애기 위한 정규화
    return ''.join(ch for ch in s.lower() if ch.isalnum())

def extract_category_from_seq_name(seq_name: str) -> str:
    """
    보통 'sub17_woodchair_053' 형태.
    -> 중간 토큰(들)을 category로 간주: 'woodchair'
       (multi-token이면 join해서 처리)
    """
    toks = seq_name.split('_')
    if len(toks) >= 3:
        cat_tokens = toks[1:-1]
        cat = '_'.join(cat_tokens) if cat_tokens else toks[1]
    elif len(toks) >= 2:
        cat = toks[1]
    else:
        cat = seq_name
    return cat

def jpos22_from_motion(motion: np.ndarray, keep_idx_csv: str | None) -> np.ndarray:
    """
    motion: (T, 24*3 + 24*3 + 22*6)
    -> jpos24: (T,24,3) -> jpos22: (T,22,3)
    """
    T = motion.shape[0]
    assert motion.shape[1] >= 24*3, f"motion feature dim too small: {motion.shape}"
    jpos24 = motion[:, :24*3].reshape(T, 24, 3)

    if keep_idx_csv:
        keep_idx = [int(x) for x in keep_idx_csv.split(',')]
        assert len(keep_idx) == 22, f"keep_idx must have 22 ints, got {len(keep_idx)}"
    else:
        # 기본: 앞 22개 사용 (필요시 --keep_idx로 정확 매핑 지정)
        keep_idx = list(range(22))

    return jpos24[:, keep_idx, :].astype(np.float32)

def decide_split_from_path(path: str, default: str = "train") -> str:
    name = path.lower()
    if "train" in name: return "train"
    if "test" in name or "val" in name: return "test"
    return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="joblib 파일 경로들 (glob 가능). 예) .../cano_train_*.p .../cano_test_*.p")
    ap.add_argument("--pose_dir", default="results/david/pose_data")
    ap.add_argument("--obj_dir", default="results/david/obj")
    ap.add_argument("--dataset", default="FullBodyManip")
    ap.add_argument("--category", required=True,
                    help="필터링할 카테고리명 (예: woodchair, largetable 등)")
    ap.add_argument("--keep_idx", default="",
                    help="24→22 인덱스 CSV (예: '0,1,2,...,21'). 비우면 앞 22개")
    ap.add_argument("--min_len", type=int, default=1,
                    help="최소 프레임 수 미만은 스킵")
    args = ap.parse_args()

    COMPATIBILITY_MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

    target_cat_norm = normalize(args.category)
    total_saved = 0
    total_seen = 0
    category = {}
    for in_pattern in args.inputs:
        for p in glob.glob(in_pattern):
            split = decide_split_from_path(p)
            print(f"[info] loading: {p} (split={split})")
            data = joblib.load(p)  # list[dict] or dict

            # 데이터 컨테이너를 통일된 방식으로 순회
            for key in data:
                seq = data[key]
                if not isinstance(seq, dict) or "motion" not in seq or "seq_name" not in seq:
                    continue
                if seq['seq_name'] not in ['sub2_largetable_003', 'sub2_largetable_005', 'sub2_largetable_012']:
                    continue
                total_seen += 1
                seq_name = str(seq["seq_name"])
                seq_cat   = extract_category_from_seq_name(seq_name)
                category[seq_cat] = 1
                if normalize(seq_cat) != target_cat_norm:
                    continue

                motion = np.asarray(seq["motion"])
                if motion.ndim != 2:
                    print(f"[skip] {seq_name}: motion ndim={motion.ndim}")
                    continue

                if motion.shape[0] < 10:
                    print(f"[skip] {seq_name}: motion length={motion.shape[0]}")
                    continue

                
                if not os.path.exists(f'data/omomo_text_anno_json_data/{seq_name}.json'):
                    print(f"[skip] {seq_name}: no text description key:{key}")
                    continue
                with open(f'data/omomo_text_anno_json_data/{seq_name}.json', 'r') as f:
                    text_description = json.load(f)[seq_name]

                try:
                    diff = seq['window_obj_com_pos'][1:] - seq['window_obj_com_pos'][:-1]
                    diff_square = np.sum(diff**2, axis=1, keepdims=True)
                    diff_mask = diff_square < 1e-05
                    m = diff_mask.ravel()          # (T-1,)
                    false_idx = np.flatnonzero(~m) # False인 위치들
                    t1 = int(false_idx[0]) if false_idx.size else None
                    t2 = int(false_idx[-1]) if false_idx.size else None
                    if t1 is None or t2 - t1 < 10:
                        print(f"[skip] {seq_name}: t1 - {t1}, t2 - {t2}")
                        continue
                    t1 = 0
                    t2 = motion.shape[0]
                    jpos22 = jpos22_from_motion(motion[t1:t2, ...], args.keep_idx if args.keep_idx else None)
                    jpos22 = COMPATIBILITY_MATRIX @ jpos22.astype(np.float32)[..., None]
                    jpos22 = jpos22[..., 0]
                except AssertionError as e:
                    print(f"[skip] {seq_name}: {e}")
                    continue
                if jpos22.shape[0] < args.min_len:
                    print(f"[skip] {seq_name}: T={jpos22.shape[0]} < min_len={args.min_len}")
                    continue

                # 저장 경로: {pose_dir}/FullBodyManip/{category}/{split}/{seq_name}/000000.npy
                out_dir = Path(args.pose_dir) / args.dataset / args.category / split / f'{seq_name}_{key}'
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "000000.npy"
                np.save(out_path.as_posix(), jpos22)

                # get pose using SMPLify
                npy_data = {
                    "motion": np.transpose(jpos22, (1, 2, 0))[None],    # [T, 22, 3] -> [1, 22, 3, T]
                    "text": [text_description],
                    "lengths": np.array([jpos22.shape[0]], dtype=int),
                    "num_samples": 1,
                    "num_repetitions": 1,
                }
                npy_path = out_dir / "000000_mdm.npy"
                np.save(npy_path.as_posix(), npy_data)
                smplify = vis_utils.npy2obj(npy_path.as_posix(), 0, 0, opt_beta=False, device=0, cuda=True)
                smplify_data = smplify.get_npy()
                # smply_data:
                # motion (25, 6, 98)
                # thetas (24, 6, 98)
                # root_translation (3, 98)
                # faces (13776, 3)
                # vertices torch.Size([6890, 3, 98])
                # text Lift the largetable, move the largetable and put down the largetable.
                # length 98
                # opt_dict
                #         pose torch.Size([98, 72])
                #         betas torch.Size([98, 10])
                #         cam torch.Size([98, 1, 3])

                # object data
                obj_trans = COMPATIBILITY_MATRIX @ seq["window_obj_com_pos"][t1:t2, ..., None]    # [T, 3, 1]
                obj_rot = COMPATIBILITY_MATRIX @ seq["obj_rot_mat"][t1:t2]    # [T, 3, 3]
                trans = COMPATIBILITY_MATRIX @ seq["global_root_trans"][t1:t2, ..., None]   # [T, 3, 1]
                trans = trans[..., 0]

                # plot
                obj_verts = get_object_vertices(torch.from_numpy(obj_trans[..., 0]).to(dtype=torch.float32), torch.from_numpy(obj_rot).to(dtype=torch.float32), 0.15)
                plot_3d_points(f'{seq_name}_{t1}_{t2}.mp4', jpos22, obj_verts, title="jpos22", dataset='humanml', fps=30)

                # root trans 비교
                print('trans loss', np.sum(np.abs(smplify_data['root_translation'].T - trans)))

                # save object data
                obj_dir = Path(args.obj_dir) / args.dataset / args.category / split / f'{seq_name}_{key}'
                obj_dir.mkdir(parents=True, exist_ok=True)
                obj_path = obj_dir / "000000.npz"
                np.savez(
                    obj_path.as_posix(), 
                    obj_trans=obj_trans, 
                    obj_rot=obj_rot, 
                    poses=smplify_data['opt_dict']['pose'].detach().cpu().numpy(), # [T, 72]
                    trans=smplify_data['root_translation'].T, # [T, 3]
                    betas=smplify_data['opt_dict']['betas'].detach().cpu().numpy(),  # [T, 10]
                )
                print(smplify_data['opt_dict']['betas'].detach().cpu().numpy())

                total_saved += 1
                if total_saved % 50 == 0:
                    print(f"[info] saved {total_saved} sequences ...")
                    

    print(f"[done] seen={total_seen}, saved={total_saved}")

if __name__ == "__main__":
    main()
