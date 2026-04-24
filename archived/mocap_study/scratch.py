from rl.mocap_study.envs.mocap_reference import MocapReference
from rl.mocap_study.envs.reward_tracking import build_tracking_indices, compute_tracking_reward

ref = MocapReference(
    "rl/mocap_study/output/reference/trial0_normalized_cycles_with_phase.csv",
    cycle_id=0,
)

track_idx = build_tracking_indices(ref.pos_cols)

a = ref.get(0.10)
b = ref.get(0.12)

out = compute_tracking_reward(
    q=a["pos"],
    dq=a["vel"],
    q_ref=b["pos"],
    dq_ref=b["vel"],
    track_idx=track_idx,
)

print("track_idx:", track_idx)
print(out)
