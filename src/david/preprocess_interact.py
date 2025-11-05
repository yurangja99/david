import os, argparse, glob, joblib, numpy as np
from pathlib import Path
import json
import torch
from imports.mdm.visualize import vis_utils
from imports.mdm.data_loaders.humanml.utils import paramUtil
from utils.visualize import plot_3d_points, get_object_vertices
from pytorch3d.transforms import quaternion_to_matrix
SKELETON = paramUtil.t2m_kinematic_chain + paramUtil.t2m_left_hand_chain + paramUtil.t2m_right_hand_chain

def normalize(s: str) -> str:
    # 대소문자/언더스코어/하이픈/띄어쓰기 차이를 없애기 위한 정규화
    s = s.split("_")[0]
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

def jpos22_from_motion(motion: np.ndarray) -> np.ndarray:
    """
    motion: (T, 24*3 + 24*3 + 22*6)
    -> jpos24: (T,24,3) -> jpos22: (T,22,3)
    """
    T = motion.shape[0]
    assert motion.shape[1] >= 24*3, f"motion feature dim too small: {motion.shape}"
    jpos24 = motion[:, :24*3].reshape(T, 24, 3)
    return jpos24[:, :22, :].astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data/InterAct", help="InterAct 데이터셋 폴더 주소")
    ap.add_argument("--pose_dir", default="results/david/pose_data")
    ap.add_argument("--obj_dir", default="results/david/obj_data")
    ap.add_argument("--dataset", default="FullBodyManip")
    ap.add_argument("--category", default="largetable")
    ap.add_argument("--min_len", type=int, default=1, help="최소 프레임 수 미만은 스킵")
    ap.add_argument("--crop_start", type=int, default=0, help="HOI 시작 부분 자를 프레임 수")
    ap.add_argument("--crop_end", type=int, default=0, help="HOI 마지막 부분 자를 프레임 수")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    COMPATIBILITY_MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

    target_cat_norm = normalize(args.category)
    total_saved = 0
    total_seen = 0
    category = {}
    for p in os.listdir(os.path.join(args.input_dir, args.category)):
        data = torch.load(os.path.join(args.input_dir, args.category, p))

        seq_name = p.replace(".pt", "")
        split = "train"
        total_seen += 1

        cat = extract_category_from_seq_name(seq_name)
        category[cat] = 1
        
        if normalize(cat) != target_cat_norm:
            print(f"[skip] {seq_name}: category mismatch: {normalize(cat)} != {target_cat_norm}")
            continue
        if data.ndim != 2:
            print(f"[skip] {seq_name}: data ndim={data.ndim}")
            continue
        if data.shape[0] < 10:
            print(f"[skip] {seq_name}: data length={data.shape[0]}")
            continue
        if not os.path.exists(f'data/omomo_text_anno_json_data/{seq_name}.json'):
            print(f"[skip] {seq_name}: no text description")
            continue
        with open(f'data/omomo_text_anno_json_data/{seq_name}.json', 'r') as f:
            text_description = json.load(f)[seq_name]

        # crop non-interactive frames
        contact_mask = data[:, 330] > 0.5
        contact_idx = torch.nonzero(contact_mask.flatten(), as_tuple=True)[0]
        t1 = int(contact_idx[0]) + args.crop_start if contact_idx.shape[0] > 0 else None
        t2 = int(contact_idx[-1]) - args.crop_end if contact_idx.shape[0] > 0 else None
        if t1 is None or t2 - t1 < 10:
            print(f"[skip] {seq_name}: t1={t1}, t2={t2}")
            continue
        else:
            print(f"[info] {seq_name}: t1={t1}, t2={t2}")
        
        data = data[t1:t2]
        
        # interpolate: 30FPS -> 20FPS
        data = torch.nn.functional.interpolate(
            data.transpose(0, 1).unsqueeze(0),  # [1, 591, T]
            size=int(round(data.shape[0] * 20 / 30)),
            mode="linear",
            align_corners=True,
        ).squeeze(0).transpose(0, 1)

        # get and save joint pos
        jpos52 = data[:, 162:162+52*3].reshape(-1, 52, 3).detach().cpu().numpy().astype(np.float32)
        jpos22 = jpos52[:, [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 33, 13, 15, 34, 16, 35, 17, 36], :]
        jpos22 = COMPATIBILITY_MATRIX @ jpos22[..., None]
        jpos22 = jpos22[..., 0]
        out_dir = Path(args.pose_dir) / args.dataset / args.category / split / f'{seq_name}'
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
        smplify = vis_utils.npy2obj(npy_path.as_posix(), 0, 0, device=args.device, cuda=True)
        smplify_data = smplify.get_npy()
        os.remove(npy_path)

        # get and save obj
        obj_trans = data[:, 318:321].detach().cpu().numpy()
        obj_quat = data[:, [324, 321, 322, 323]]    # wxyz type
        obj_rot = quaternion_to_matrix(obj_quat).detach().cpu().numpy()
        
        obj_trans = COMPATIBILITY_MATRIX @ obj_trans[..., None]
        obj_trans = obj_trans[..., 0]
        # obj_rot = (COMPATIBILITY_MATRIX @ (obj_rot @ ROT_OFS) @ COMPATIBILITY_MATRIX.T)
        NEW_ROT_OFS = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])    # x축 기준 -90도 회전
        obj_rot = NEW_ROT_OFS @ obj_rot
        
        obj_dir = Path(args.obj_dir) / args.dataset / args.category / split / f'{seq_name}'
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

        # plot results
        obj_vertices = get_object_vertices(
            torch.from_numpy(obj_trans).to(torch.float32), 
            torch.from_numpy(obj_rot).to(torch.float32), 
            object=target_cat_norm,
        ).detach().cpu().numpy()
        ani_path = obj_dir / "000000.mp4"
        plot_3d_points(
            ani_path.as_posix(), 
            SKELETON, jpos22, 
            target_cat_norm, obj_vertices, 
            title="Joints and Object", 
            dataset="humanml", fps=20, show_joints=False
        )

        total_saved += 1
        if total_saved % 50 == 0:
            print(f"[info] saved {total_saved} sequences ...")

    print(f"[done] seen={total_seen}, saved={total_saved}")

if __name__ == "__main__":
    main()
