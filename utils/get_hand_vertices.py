import numpy as np
import torch
from smplx import SMPLX
import smplx
import pickle

def collect_descendants(parents_list, root_idx):
    """parents_list: 길이 J의 리스트/배열 (parents[j] = j의 부모 인덱스, 루트는 -1)"""
    children = {i: [] for i in range(len(parents_list))}
    for j, p in enumerate(parents_list):
        if p >= 0:
            children[p].append(j)
    stack = [root_idx]
    out = set([root_idx])
    while stack:
        cur = stack.pop()
        for c in children[cur]:
            if c not in out:
                out.add(c)
                stack.append(c)
    return sorted(out)

def get_parents_vector(model):
    """SMPL-X 빌드별 parents 포맷(텐서/ndarray/리스트)을 일관된 1D 리스트로 변환"""
    p = getattr(model, "parents", None)
    if p is None:
        # 매우 오래된 빌드 대응 (거의 안 씀)
        kt = getattr(model, "kintree_table", None)
        if kt is not None:
            p = kt[0]
        else:
            raise RuntimeError("No parents/kintree_table found in model.")
    if torch.is_tensor(p):
        return p.view(-1).cpu().tolist()
    else:
        arr = np.array(p).reshape(-1)
        return arr.tolist()

def get_lbs_weights_VJ(model):
    """(V,J) 형태의 LBS 가중치 반환 (모델마다 (1,V,J)인 경우가 많음)"""
    W = getattr(model, "lbs_weights", None)
    if W is None:
        raise RuntimeError("Model has no lbs_weights.")
    if torch.is_tensor(W):
        if W.dim() == 3:
            W = W[0]       # (1,V,J) → (V,J)
        elif W.dim() == 2:
            W = W
        else:
            raise RuntimeError(f"Unexpected lbs_weights dim: {W.dim()}")
        return W
    else:
        W = np.array(W)
        if W.ndim == 3:
            W = W[0]
        if not torch.is_tensor(W):
            W = torch.tensor(W)
        return W

def find_first_idx_by_name(joint_names, candidates):
    for i, name in enumerate(joint_names):
        for c in candidates:
            if c in name:
                return i
    return None

# ───────────── 실행 파트 ─────────────
model = smplx.create(model_path="imports/mdm/body_models/smplx/SMPLX_NEUTRAL.npz", model_type="smplx", num_pca_comps=45).cpu()

parents = get_parents_vector(model)     # ★ 여기서 더이상 [0] 인덱싱 금지
W = get_lbs_weights_VJ(model)           # (V,J) torch.Tensor

joint_names = getattr(model, "joint_names", None)

if joint_names is not None:
    L_WRIST = find_first_idx_by_name(joint_names, ["left_wrist", "l_wrist"])
    R_WRIST = find_first_idx_by_name(joint_names, ["right_wrist", "r_wrist"])
else:
    # 공동 작업용 안전 기본값(빌드마다 다를 수 있음→ 필요시 맞춰 수정)
    L_WRIST, R_WRIST = 20, 21

assert L_WRIST is not None and R_WRIST is not None, "손목 조인트를 찾을 수 없습니다."

L_HAND_JOINTS = collect_descendants(parents, L_WRIST)
R_HAND_JOINTS = collect_descendants(parents, R_WRIST)
print(L_HAND_JOINTS)
print(R_HAND_JOINTS)

argmaxJ = W.argmax(dim=1).cpu().numpy()           # 각 버텍스가 가장 의존하는 조인트
left_hand_verts  = np.where(np.isin(argmaxJ, L_HAND_JOINTS))[0]
right_hand_verts = np.where(np.isin(argmaxJ, R_HAND_JOINTS))[0]

print(f"#left hand verts: {left_hand_verts}")
print(f"#right hand verts: {right_hand_verts}")
print(f"#left hand verts: {len(left_hand_verts)}")
print(f"#right hand verts: {len(right_hand_verts)}")

with open("hand_verts.pkl", "wb") as f:
  pickle.dump(left_hand_verts.tolist()+right_hand_verts.tolist(), f)
# np.save("smplx_left_hand_verts.npy", left_hand_verts)
# np.save("smplx_right_hand_verts.npy", right_hand_verts)
