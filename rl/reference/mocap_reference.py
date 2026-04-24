"""Phase-indexed access to the derived mocap reference snapshot used in Stage 2."""

from __future__ import annotations

import pandas as pd


class MocapReference:
    def __init__(self, csv_path: str, cycle_id: int = 0):
        df = pd.read_csv(csv_path)
        ref = df[df["cycle_id"] == cycle_id].reset_index(drop=True)

        if len(ref) == 0:
            raise ValueError(f"No rows found for cycle_id={cycle_id}")

        self.phase = ref["phase"].to_numpy()
        self.pos_cols = [col for col in ref.columns if col.startswith("pos_")]
        self.vel_cols = [col for col in ref.columns if col.startswith("vel_")]
        self.pos = ref[self.pos_cols].to_numpy()
        self.vel = ref[self.vel_cols].to_numpy()
        self.n = len(ref)

    def get(self, phase: float) -> dict:
        phase = float(phase) % 1.0
        idx = int(round(phase * (self.n - 1)))
        idx = min(max(idx, 0), self.n - 1)

        return {
            "phase": phase,
            "pos": self.pos[idx],
            "vel": self.vel[idx],
            "pos_cols": self.pos_cols,
            "vel_cols": self.vel_cols,
            "index": idx,
        }
