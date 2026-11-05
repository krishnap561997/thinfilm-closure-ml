import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, style as S
NPZ = S.CLOSURE / "mesh_study/compare/aligned_profiles.npz"
def aligned(): return np.load(NPZ)
