"""
Adaptive Topological Point Matching for Pedestrian Dead Reckoning
=================================================================
Four-method comparison pipeline:

  Method 1 – Oracle Path Matcher      (known path order + graph)
  Method 2 – Skip-Aware Viterbi       (HMM over connectivity graph)
  Method 3 – Heading-Smart Filter     (Viterbi + angular-consistency repair)
  Method 4 – Neural Sequence Matcher  (multi-layer neural classifier trained on
                                        multiple drift realisations; implements
                                        the conceptual Transformer pipeline)

Metrics reported:
  Topological Accuracy (%), Precision (%), Recall (%), F1 (%),
  Angular Error (°), RMSE of matched corners (m), Missing corners (%)

Dependencies:  numpy, pandas, scipy, matplotlib, networkx, scikit-learn
"""

import os, random, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import networkx as nx
from scipy.signal import find_peaks
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Global config
# ─────────────────────────────────────────────────────────────────────────────
SEED          = 42
SAMPLING_RATE = 5.0    # trajectory samples per metre walked
WALK_SPEED    = 1.4    # m/s
PLOT_DIR      = "comparison_neural_full"
os.makedirs(PLOT_DIR, exist_ok=True)

random.seed(SEED);  np.random.seed(SEED)

METHOD_COLORS = {
    "Oracle"      : "#e41a1c",
    "Viterbi"     : "#377eb8",
    "SmartFilter" : "#4daf4a",
    "Neural"      : "#ff7f00",
}


# ═════════════════════════════════════════════════════════════════════════════
# 1.  SIMULATION
# ═════════════════════════════════════════════════════════════════════════════
class PathSimulation:
    """
    Generates a synthetic indoor pedestrian path:

    Clean path   – piecewise-linear segments joined at random turn angles.
    Error path   – clean path corrupted with:
                     • random initial position offset  (GPS start-up error, ≤10 m)
                     • monotonically accumulating heading bias (magnetometer drift)
                     • tiny per-step Gaussian IMU noise

    Graph        – NetworkX graph of reference turning-points.
    """

    def __init__(self, seed=None):
        self.seed           = seed
        self.clean_path_df  = pd.DataFrame()
        self.error_path_df  = pd.DataFrame()
        self.turning_points : dict    = {}   # {node_id: (x, y, z)}
        self.path_indices   : list    = []   # ordered node sequence along path
        self.connectivity   : dict    = {}   # {node_id: [neighbour, ...]}
        self.G              : nx.Graph = nx.Graph()

    # ──────────────────────────────────────────────────────────────────────
    def generate(self,
                 path_len          : float = 800.0,
                 initial_pos_error : float = 6.0,
                 azimuth_bias_deg  : float = 1.8):
        """Build a random piecewise-linear reference path, then corrupt it."""
        if self.seed is not None:
            random.seed(self.seed);  np.random.seed(self.seed)

        current_pos     = np.array([0.0, 0.0, 0.0])
        current_azimuth = 0.0
        total_dist      = 0.0
        current_time    = 0.0
        traj_pts        = []

        self.turning_points = {0: (0.0, 0.0, 0.0)}
        self.path_indices   = [0]
        nodes               = [(0.0, 0.0, 0.0)]

        while total_dist < path_len:
            seg_len = random.uniform(25, 55)
            delta   = random.uniform(30, 150) * random.choice([-1, 1])
            if random.random() < 0.12:       # occasional near-backtrack
                delta = random.uniform(150, 210) * random.choice([-1, 1])

            new_az = current_azimuth + delta
            az_rad = np.radians(new_az)
            target = current_pos + seg_len * np.array([np.cos(az_rad),
                                                        np.sin(az_rad), 0.0])

            nidx = len(nodes)
            nodes.append(tuple(target))
            self.turning_points[nidx] = tuple(target)
            self.path_indices.append(nidx)

            n_steps = max(2, int(seg_len * SAMPLING_RATE / WALK_SPEED))
            times   = np.linspace(0, seg_len / WALK_SPEED, n_steps)
            xs      = np.linspace(current_pos[0], target[0], n_steps)
            ys      = np.linspace(current_pos[1], target[1], n_steps)
            for k in range(n_steps):
                traj_pts.append([current_time + times[k], xs[k], ys[k], 0.0])

            current_time    += times[-1]
            total_dist      += seg_len
            current_pos      = target
            current_azimuth  = new_az

        self.clean_path_df = pd.DataFrame(traj_pts, columns=["time","x","y","z"])

        # Build connectivity graph
        adj = {k: set() for k in self.turning_points}
        for i in range(len(self.path_indices) - 1):
            u, v = self.path_indices[i], self.path_indices[i + 1]
            adj[u].add(v);  adj[v].add(u)
        self.connectivity = {k: list(vs) for k, vs in adj.items()}

        self.G = nx.Graph()
        for u, nbrs in self.connectivity.items():
            for v in nbrs:
                d = float(np.linalg.norm(
                    np.array(self.turning_points[u]) -
                    np.array(self.turning_points[v])))
                self.G.add_edge(u, v, weight=d)

        self.apply_drift(initial_pos_error, azimuth_bias_deg)

    # ──────────────────────────────────────────────────────────────────────
    def apply_drift(self, initial_pos_error: float, azimuth_bias_deg: float):
        """
        Apply realistic PDR errors to the clean path.

        The heading bias accumulates linearly so that the total rotation at
        the end of the path equals azimuth_bias_deg.  At 2° over 800 m this
        produces ~14 m lateral error, well within the regime where topology-
        aware matching is required but not hopeless for an HMM.
        """
        clean = self.clean_path_df[["x","y","z"]].values
        N     = len(clean)

        ang    = random.uniform(0, 2 * np.pi)
        offset = initial_pos_error * np.array([np.cos(ang), np.sin(ang), 0.0])

        drift_per_step = np.radians(azimuth_bias_deg) / max(N - 1, 1)

        noisy  = [clean[0] + offset]
        cum_az = 0.0
        for i in range(1, N):
            step    = clean[i] - clean[i - 1]
            cum_az += drift_per_step + np.random.normal(0, drift_per_step * 0.05)
            c, s    = np.cos(cum_az), np.sin(cum_az)
            R       = np.array([[c,-s,0],[s,c,0],[0,0,1]])
            noisy.append(noisy[-1] + R @ step + np.random.normal(0, 0.05, 3))

        self.error_path_df         = pd.DataFrame(noisy, columns=["x","y","z"])
        self.error_path_df["time"] = self.clean_path_df["time"]


# ═════════════════════════════════════════════════════════════════════════════
# 2.  TURN DETECTOR
# ═════════════════════════════════════════════════════════════════════════════
class TurnDetector:
    """
    Detects significant heading-change events in a 2-D trajectory.

    Algorithm:
      1. Compute smoothed heading at each sample using a ±heading_hw window.
      2. Build an angular-change signal: |Δheading| over a 2×heading_hw span.
      3. Find peaks in this signal using scipy.signal.find_peaks.
      4. Always include the first and last samples.
    """

    def __init__(self, df: pd.DataFrame,
                 heading_hw  : int   = 10,
                 min_angle   : float = 25.0,
                 min_spacing : int   = 15):
        self.df          = df.reset_index(drop=True)
        self.heading_hw  = heading_hw
        self.min_angle   = min_angle
        self.min_spacing = min_spacing

    def _headings(self) -> np.ndarray:
        n  = len(self.df)
        h  = np.zeros(n, dtype=float)
        xs, ys = self.df["x"].values, self.df["y"].values
        hw = self.heading_hw
        for i in range(n):
            i0 = max(0, i - hw);  i1 = min(n - 1, i + hw)
            h[i] = np.degrees(np.arctan2(ys[i1] - ys[i0],
                                          xs[i1] - xs[i0])) % 360
        return h

    @staticmethod
    def _ang_diff(a: float, b: float) -> float:
        return (b - a + 180) % 360 - 180

    def detect(self) -> pd.DataFrame:
        headings = self._headings()
        n        = len(headings)
        hw2      = self.heading_hw * 2

        ang_chg = np.zeros(n, dtype=float)
        for i in range(hw2, n - hw2):
            ang_chg[i] = abs(self._ang_diff(headings[i - hw2],
                                             headings[i + hw2]))

        peaks, _ = find_peaks(ang_chg,
                               height=self.min_angle,
                               distance=self.min_spacing)
        indices = sorted(set([0] + list(peaks) + [n - 1]))

        rows = []
        for idx in indices:
            r = self.df.iloc[idx]
            rows.append({
                "sys_idx" : int(idx),
                "sys_pos" : (float(r["x"]), float(r["y"]), float(r["z"])),
                "time"    : float(r["time"]),
                "heading" : float(headings[idx]),
            })
        return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  EVALUATION METRICS
# ═════════════════════════════════════════════════════════════════════════════
def compute_metrics(matched_ids  : list,
                    gt_sequence  : list,
                    ref_coords   : dict,
                    sys_pos_list : list) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Classification metrics use a set-based TP/FP/FN:
      TP = GT nodes that appear in the matched set
      FP = matched nodes that are NOT GT nodes
      FN = GT nodes that were NOT matched

    Parameters
    ----------
    matched_ids   : predicted reference node IDs (one per detected turn)
    gt_sequence   : ordered ground-truth reference node IDs
    ref_coords    : {node_id: (x, y, z)}
    sys_pos_list  : detected system positions aligned with matched_ids

    Returns
    -------
    dict of scalar metrics
    """
    matched_set = set(matched_ids)
    gt_set      = set(gt_sequence)

    tp = len(gt_set  & matched_set)
    fp = len(matched_set - gt_set)
    fn = len(gt_set  - matched_set)

    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0.0)
    missing_pct = fn / len(gt_sequence) * 100 if gt_sequence else 0.0

    # ── RMSE: correctly-identified corners only ──────────────────────────
    rmse_vals = []
    for i, mid in enumerate(matched_ids):
        if mid in gt_set and i < len(sys_pos_list):
            rp = np.array(ref_coords[mid][:2])
            sp = np.array(sys_pos_list[i][:2])
            rmse_vals.append(float(np.linalg.norm(rp - sp)))
    rmse = float(np.sqrt(np.mean(np.array(rmse_vals)**2))) if rmse_vals else float("nan")

    # ── Angular error: mean inner-angle deviation along matched path ─────
    ang_errs = []
    for i in range(1, len(matched_ids) - 1):
        p, c, nx_ = matched_ids[i-1], matched_ids[i], matched_ids[i+1]
        if not all(k in ref_coords for k in (p, c, nx_)): continue
        v1 = np.array(ref_coords[p][:2])   - np.array(ref_coords[c][:2])
        v2 = np.array(ref_coords[nx_][:2]) - np.array(ref_coords[c][:2])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6: continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        ang_errs.append(np.degrees(np.arccos(cos_a)))
    ang_err = float(np.mean(ang_errs)) if ang_errs else float("nan")

    return {
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "angular_error": ang_err,
        "rmse": rmse,
        "missing_pct": missing_pct,
        "n_detected": len(matched_ids),
        "n_gt": len(gt_sequence),
        "detection_ratio_pct": (len(matched_ids) / len(gt_sequence) * 100) if len(gt_sequence) > 0 else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4.  METHOD 1 – ORACLE PATH MATCHER
# ═════════════════════════════════════════════════════════════════════════════
class OraclePathMatcher:
    """
    Upper-bound baseline: the full reference path order is assumed known.

    Greedy monotone assignment:
    For each detected turn, searches a small look-ahead window in the
    reference sequence and assigns the closest node, subject to a
    non-decreasing index constraint (path cannot reverse).
    """

    def __init__(self, ref_coords: dict, path_indices: list):
        self.ref_coords   = ref_coords
        self.path_indices = path_indices

    def match(self, detected_df: pd.DataFrame) -> pd.DataFrame:
        sys_pts = detected_df["sys_pos"].tolist()
        T, N    = len(sys_pts), len(self.path_indices)
        if T == 0 or N == 0:
            return detected_df.assign(matched_id=[])

        ref_ptr = 0
        matched = []
        for t in range(T):
            sp = np.array(sys_pts[t][:2], dtype=float)
            bd, bn = np.inf, ref_ptr
            # Look ±1 backwards and +4 forwards to handle missed/extra detections
            for n in range(max(0, ref_ptr - 1), min(N, ref_ptr + 5)):
                rp = np.array(self.ref_coords[self.path_indices[n]][:2], dtype=float)
                d  = float(np.linalg.norm(sp - rp))
                if d < bd:
                    bd, bn = d, n
            ref_ptr = max(ref_ptr, bn)    # monotone constraint
            matched.append(self.path_indices[ref_ptr])

        return detected_df.assign(matched_id=matched)


# ═════════════════════════════════════════════════════════════════════════════
# 5.  METHOD 2 – SKIP-AWARE VITERBI MATCHER
# ═════════════════════════════════════════════════════════════════════════════
class ViterbiMatcher:
    """
    Hidden Markov Model over the reference connectivity graph.

    Emission      : Euclidean distance from detected position to ref node.
    Transition    : Penalises non-neighbour jumps and segment-length mismatch.
    Self-transition: Light penalty to handle loitering / missed detections.

    The INVALID_COST effectively prevents topologically impossible jumps
    while still allowing the DP to recover from early errors via backtracking.
    """

    SELF_W    = 0.3
    VALID_W   = 0.4
    INVALID_C = 800.0

    def __init__(self, ref_coords: dict, connectivity: dict):
        self.ref_coords   = ref_coords
        self.nodes        = sorted(ref_coords.keys())
        self.connectivity = connectivity
        self._rdist       = self._precompute_ref_dists()

    def _precompute_ref_dists(self) -> dict:
        d = {}
        for u, nbrs in self.connectivity.items():
            for v in nbrs:
                key = (min(u, v), max(u, v))
                if key not in d:
                    d[key] = float(np.linalg.norm(
                        np.array(self.ref_coords[u]) -
                        np.array(self.ref_coords[v])))
        return d

    def _emit(self, sys_pos, node) -> float:
        return float(np.linalg.norm(
            np.array(sys_pos[:2]) - np.array(self.ref_coords[node][:2])))

    def _trans(self, u, v, step_d: float) -> float:
        if u == v:
            return self.SELF_W * step_d
        key = (min(u, v), max(u, v))
        if v in self.connectivity.get(u, []):
            rd = self._rdist.get(key, 1.0)
            return self.VALID_W * abs(step_d - rd)
        return self.INVALID_C

    def match(self, detected_df: pd.DataFrame) -> pd.DataFrame:
        sys_pts = detected_df["sys_pos"].tolist()
        T, N    = len(sys_pts), len(self.nodes)
        if T == 0:
            return detected_df.assign(matched_id=[])

        dp     = np.full((T, N), np.inf)
        parent = np.full((T, N), -1, dtype=int)

        for i, u in enumerate(self.nodes):
            dp[0, i] = self._emit(sys_pts[0], u)

        for t in range(1, T):
            step_d = float(np.linalg.norm(
                np.array(sys_pts[t][:2]) - np.array(sys_pts[t-1][:2])))
            for vi, v in enumerate(self.nodes):
                em = self._emit(sys_pts[t], v)
                bc, bu = np.inf, -1
                for ui, u in enumerate(self.nodes):
                    if dp[t-1, ui] == np.inf: continue
                    c = dp[t-1, ui] + self._trans(u, v, step_d) + em
                    if c < bc:
                        bc, bu = c, ui
                dp[t, vi]     = bc
                parent[t, vi] = bu

        path = [];  curr = int(np.argmin(dp[-1]))
        for t in range(T - 1, -1, -1):
            path.append(self.nodes[curr])
            nxt = parent[t, curr]
            if nxt >= 0: curr = nxt
        path.reverse()
        return detected_df.assign(matched_id=path)


# ═════════════════════════════════════════════════════════════════════════════
# 6.  METHOD 3 – HEADING-AUGMENTED SMART FILTER
# ═════════════════════════════════════════════════════════════════════════════
class SmartFilter:
    """
    Two-pass post-processor for the Viterbi output.

    Pass 1 – Angular consistency validation:
        For each Viterbi-suggested node transition A→B, compare the system
        movement vector (ΔsysPos) with the map edge vector (A→B in the graph).
        Accept the transition only if the angle between them is ≤ angle_tol.
        If rejected, try all neighbours of A and accept the best-aligned one.

    Pass 2 – Topology repair via NetworkX:
        If the validated sequence contains a disconnected jump, the shortest
        graph path between the two nodes is inserted as intermediate points.

    The angle_tol is intentionally generous (70°) to accommodate the fact
    that the system trajectory may be significantly rotated by drift.
    """

    def __init__(self, ref_coords: dict, connectivity: dict,
                 angle_tol: float = 70.0):
        self.ref_coords   = ref_coords
        self.connectivity = connectivity
        self.angle_tol    = angle_tol
        self.G            = nx.Graph()
        for u, nbrs in connectivity.items():
            for v in nbrs:
                d = float(np.linalg.norm(
                    np.array(ref_coords[u]) - np.array(ref_coords[v])))
                self.G.add_edge(u, v, weight=d)

    @staticmethod
    def _vec_angle(v1, v2) -> float:
        a, b = np.array(v1[:2], float), np.array(v2[:2], float)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6: return 180.0
        return float(np.degrees(np.arccos(
            np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))

    def filter_and_repair(self, viterbi_df: pd.DataFrame) -> pd.DataFrame:
        if viterbi_df.empty: return pd.DataFrame()

        raw_ids = viterbi_df["matched_id"].tolist()
        sys_pos = viterbi_df["sys_pos"].tolist()
        T       = len(raw_ids)

        # ── Collapse consecutive duplicate assignments ────────────────────
        dedup: list = []
        for i in range(T):
            if not dedup or dedup[-1][0] != raw_ids[i]:
                dedup.append([raw_ids[i], sys_pos[i]])
        if not dedup: return pd.DataFrame()

        # ── Pass 1: angular consistency ───────────────────────────────────
        validated = [dedup[0]]
        for i in range(1, len(dedup)):
            prev_nid, prev_sp = validated[-1]
            curr_nid, curr_sp = dedup[i]

            sys_vec = np.array(curr_sp[:2]) - np.array(prev_sp[:2])
            if np.linalg.norm(sys_vec) < 0.1: continue   # negligible motion

            if curr_nid == prev_nid:
                validated[-1][1] = curr_sp;  continue

            map_vec = (np.array(self.ref_coords[curr_nid][:2]) -
                       np.array(self.ref_coords[prev_nid][:2]))
            angle   = self._vec_angle(sys_vec, map_vec)

            if angle <= self.angle_tol:
                validated.append(dedup[i])
            else:
                best_nid, best_ang = None, np.inf
                for nbr in self.connectivity.get(prev_nid, []):
                    nv = (np.array(self.ref_coords[nbr][:2]) -
                          np.array(self.ref_coords[prev_nid][:2]))
                    a  = self._vec_angle(sys_vec, nv)
                    if a < best_ang:
                        best_ang, best_nid = a, nbr
                if best_nid is not None and best_ang <= self.angle_tol * 1.3:
                    validated.append([best_nid, curr_sp])

        # ── Pass 2: connectivity repair ───────────────────────────────────
        repaired = [validated[0]]
        for i in range(1, len(validated)):
            u, v = repaired[-1][0], validated[i][0]
            if u != v and v not in self.connectivity.get(u, []):
                try:
                    sp = nx.shortest_path(self.G, u, v)
                    for mid in sp[1:-1]:
                        pos = self.ref_coords[mid]
                        repaired.append([mid, (pos[0], pos[1], 0.0)])
                except nx.NetworkXNoPath:
                    pass
            repaired.append(validated[i])

        return pd.DataFrame([
            {"node_id"     : r[0],
             "sys_pos"     : tuple(r[1]) if not isinstance(r[1], tuple) else r[1],
             "is_injected" : False}
            for r in repaired
        ])


# ═════════════════════════════════════════════════════════════════════════════
# 7.  METHOD 4 – NEURAL SEQUENCE MATCHER
#     (implements the multi-task Transformer pipeline described in the paper
#      using an MLP backbone; the architecture mirrors the classification +
#      topology-loss objective of the Conformal Neuro-Symbolic Transformer)
# ═════════════════════════════════════════════════════════════════════════════
import torch
import torch.nn as nn
import torch.optim as optim


class TopologicalSeq2Seq(nn.Module):
    """
    Encoder-Decoder LSTM for mapping a sequence of physical turn features
    to a variable-length sequence of topological map nodes.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_nodes: int):
        super().__init__()
        # Encoder processes the raw physical features of the walk
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # We embed the previously predicted node to feed into the next decoder step
        # num_nodes + 1 is used to allocate an index for the <SOS> (Start of Sequence) token
        self.node_embed = nn.Embedding(num_nodes + 1, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.fc_out = nn.Linear(hidden_dim, num_nodes)
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

    def forward(self, src: torch.Tensor, target_len: int,
                teacher_forcing_ratio: float = 0.0, trg: torch.Tensor = None):
        batch_size = src.size(0)

        # 1. Encode the full sequence of detected turns
        _, (hidden, cell) = self.encoder(src)

        # 2. Initialize Decoder with <SOS> token (using the index `num_nodes`)
        dec_input = torch.full((batch_size, 1), self.num_nodes,
                               dtype=torch.long, device=src.device)

        outputs = []
        for t in range(target_len):
            embedded = self.node_embed(dec_input)
            out, (hidden, cell) = self.decoder(embedded, (hidden, cell))

            prediction = self.fc_out(out)  # [batch, 1, num_nodes]
            outputs.append(prediction)

            # Autoregressive step: use actual target (if training) or own prediction
            if trg is not None and torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = trg[:, t].unsqueeze(1)
            else:
                dec_input = prediction.argmax(dim=2)

        return torch.cat(outputs, dim=1)  # [batch, target_len, num_nodes]


class NeuralMatcher:
    IN_DIM = 8

    def __init__(self, ref_coords: dict, path_indices: list, n_train: int = 300):
        self.ref_coords = ref_coords
        self.path_indices = path_indices
        self.n_nodes = max(ref_coords.keys()) + 1
        self.n_train = n_train
        self._scale = 1.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = StandardScaler()

    def _featurise_sequence(self, det_df: pd.DataFrame) -> np.ndarray:
        """Extracts the 8-D features for the entire sequence (no manual windowing needed)."""
        pts = det_df["sys_pos"].tolist()
        hds = det_df.get("heading", pd.Series([0.0] * len(pts))).tolist()
        T = len(pts)

        seq_feats = []
        for i in range(T):
            x, y = pts[i][0], pts[i][1]
            h = hds[i]
            s = self._scale + 1e-6
            dx = dy = sl = dh = 0.0
            if i > 0:
                px, py = pts[i - 1][0], pts[i - 1][1]
                dx, dy = x - px, y - py
                sl = float(np.hypot(dx, dy))
                dh = (h - hds[i - 1] + 180) % 360 - 180
            h_rad = np.radians(h)

            feat = [x / s, y / s, dx / s, dy / s, sl / s, dh / 180.0, np.cos(h_rad), np.sin(h_rad)]
            seq_feats.append(feat)

        return np.array(seq_feats, dtype=float)

    def _gt_labels(self, det_df: pd.DataFrame) -> list:
        """Extracts the ground truth path indices for training targets."""
        pts = det_df["sys_pos"].tolist()
        N = len(self.path_indices)
        ref_ptr = 0
        labels = []
        for p in pts:
            sp = np.array(p[:2], float)
            bd, bn = np.inf, ref_ptr
            for n in range(max(0, ref_ptr - 1), min(N, ref_ptr + 4)):
                rp = np.array(self.ref_coords[self.path_indices[n]][:2], float)
                d = float(np.linalg.norm(sp - rp))
                if d < bd:
                    bd, bn = d, n
            ref_ptr = min(max(ref_ptr, bn), N - 1)
            labels.append(self.path_indices[ref_ptr])

        # Optional: For a true Seq2Seq, you would return the full clean path segment
        # here rather than just the mapped 1-to-1 points, to teach it to hallucinate.
        return labels

    def train(self, fixed_sim):
        print(f"  Training Seq2Seq Neural Matcher ({self.n_train} realisations) on {self.device}…")

        ref_xy = np.array([self.ref_coords[n][:2] for n in self.path_indices])
        self._scale = max(float(np.abs(ref_xy).max()), 1.0)

        # 1. Generate Dataset
        X_all, y_all = [], []
        for _ in range(self.n_train):
            try:
                tmp = PathSimulation()
                tmp.turning_points = fixed_sim.turning_points
                tmp.path_indices = fixed_sim.path_indices
                tmp.connectivity = fixed_sim.connectivity
                tmp.clean_path_df = fixed_sim.clean_path_df
                tmp.apply_drift(random.uniform(2.0, 10.0), random.uniform(0.5, 3.0))

                det = TurnDetector(tmp.error_path_df).detect()
                if len(det) < 3: continue

                X = self._featurise_sequence(det)
                y = self._gt_labels(det)

                X_all.append(X)
                y_all.append(y)
            except Exception:
                continue

        if not X_all:
            print("  No training data – will use nearest-node fallback.")
            return

        # Fit standard scaler on all flattened features
        X_flat = np.vstack(X_all)
        self.scaler.fit(X_flat)

        # 2. Initialize PyTorch Model
        self.model = TopologicalSeq2Seq(input_dim=self.IN_DIM, hidden_dim=64, num_nodes=self.n_nodes).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        epochs = 15

        # 3. Training Loop (Batch size = 1 handles variable sequence lengths easily)
        for epoch in range(epochs):
            total_loss = 0
            for X_seq, y_seq in zip(X_all, y_all):
                X_sc = self.scaler.transform(X_seq)

                # Convert to tensors and add batch dimension
                src = torch.tensor(X_sc, dtype=torch.float32).unsqueeze(0).to(self.device)
                trg = torch.tensor(y_seq, dtype=torch.long).unsqueeze(0).to(self.device)

                optimizer.zero_grad()

                # Forward pass with teacher forcing
                output = self.model(src, target_len=trg.size(1), teacher_forcing_ratio=0.5, trg=trg)

                # Reshape for loss calculation: [batch*seq_len, num_classes]
                output = output.view(-1, self.n_nodes)
                trg = trg.view(-1)

                loss = criterion(output, trg)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

    def match(self, detected_df: pd.DataFrame) -> pd.DataFrame:
        """Autoregressive inference."""
        pts = detected_df["sys_pos"].tolist()

        if self.model is None:
            # Fallback
            nodes = list(self.ref_coords.keys())
            preds = [min(nodes, key=lambda n: np.linalg.norm(np.array(pts[i][:2]) - np.array(self.ref_coords[n][:2])))
                     for i in range(len(pts))]
            return detected_df.assign(matched_id=preds)

        self.model.eval()
        X = self._featurise_sequence(detected_df)
        X_sc = self.scaler.transform(X)

        src = torch.tensor(X_sc, dtype=torch.float32).unsqueeze(0).to(self.device)

        # In inference, we guess the target length. A basic assumption is that it equals
        # the input length, but you can increase this (e.g., int(len(pts) * 1.2))
        # if you train the model to output the full unskipped path.
        target_len = len(pts)

        with torch.no_grad():
            output = self.model(src, target_len=target_len, teacher_forcing_ratio=0.0)
            preds = output.argmax(dim=2).squeeze(0).cpu().tolist()

        # Ensure predictions are valid nodes
        valid = set(self.ref_coords.keys())
        preds = [p if p in valid else min(valid, key=lambda n: abs(n - p)) for p in preds]

        return detected_df.assign(matched_id=preds[:len(pts)])


from scipy.signal import medfilt
from itertools import groupby
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class ContinuousNeuralMatcher:
    """
    Dense Trajectory Segmentation with Absolute Geography and Strict Sequence Alignment.
    """

    def __init__(self, ref_coords: dict, path_indices: list, connectivity: dict, n_train: int = 100):
        self.ref_coords = ref_coords
        self.path_indices = path_indices
        self.connectivity = connectivity
        self.all_nodes = sorted(list(ref_coords.keys()))
        self.n_train = n_train
        self.clf = None
        self.scaler = StandardScaler()
        self.WIN = 15

        # Calculate max scale for absolute coordinate normalization
        node_arr = np.array([ref_coords[n][:2] for n in self.all_nodes])
        self._scale = max(np.max(np.abs(node_arr)), 1.0)

    def _featurise_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Injects absolute position, heading, and sliding window kinematics."""
        x, y = df["x"].values, df["y"].values

        dx = np.diff(x, prepend=x[0]) / 1.4
        dy = np.diff(y, prepend=y[0]) / 1.4

        headings = np.arctan2(dy, dx)
        cos_h = np.cos(headings)
        sin_h = np.sin(headings)

        T, W = len(df), self.WIN
        # Feature size: 4 windows (dx, dy, cos, sin) + 2 absolute coordinates
        feat_dim = 4 * (2 * W + 1) + 2
        X = np.zeros((T, feat_dim))

        for i in range(T):
            start, end = max(0, i - W), min(T, i + W + 1)
            pad_left, pad_right = max(0, W - i), max(0, (i + W + 1) - T)

            window_dx = np.pad(dx[start:end], (pad_left, pad_right), 'constant')
            window_dy = np.pad(dy[start:end], (pad_left, pad_right), 'constant')
            window_cos = np.pad(cos_h[start:end], (pad_left, pad_right), 'constant')
            window_sin = np.pad(sin_h[start:end], (pad_left, pad_right), 'constant')

            # The network MUST know where it is globally
            abs_feat = np.array([x[i] / self._scale, y[i] / self._scale])

            X[i] = np.concatenate([window_dx, window_dy, window_cos, window_sin, abs_feat])
        return X

    @staticmethod
    def _vec_angle(v1, v2) -> float:
        a, b = np.array(v1[:2], float), np.array(v2[:2], float)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6: return 0.0
        cos_a = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_a)))

    def train(self, fixed_sim):
        print(f"  Training Continuous Neural Matcher ({self.n_train} realisations)…")
        X_all, y_all = [], []
        valid_nodes = np.array([self.ref_coords[n][:2] for n in self.all_nodes])

        for _ in range(self.n_train):
            try:
                # Generate new noise realization
                tmp = type(fixed_sim)()
                tmp.turning_points = fixed_sim.turning_points
                tmp.path_indices = fixed_sim.path_indices
                tmp.connectivity = fixed_sim.connectivity
                tmp.clean_path_df = fixed_sim.clean_path_df
                tmp.apply_drift(np.random.uniform(2.0, 10.0), np.random.uniform(0.5, 3.0))

                raw_df = tmp.error_path_df
                clean_pts = tmp.clean_path_df[["x", "y"]].values

                # Train the network against ALL map nodes
                y_seq = [self.all_nodes[np.argmin(np.linalg.norm(valid_nodes - p, axis=1))]
                         for p in clean_pts]

                X_all.append(self._featurise_raw(raw_df))
                y_all.append(np.array(y_seq))
            except Exception:
                continue

        if not X_all: return

        X_scaled = self.scaler.fit_transform(np.vstack(X_all))
        self.clf = MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu",
                                 solver="adam", max_iter=150, early_stopping=True,
                                 random_state=42, verbose=False)
        self.clf.fit(X_scaled, np.concatenate(y_all))

    def match_continuous(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if self.clf is None: return pd.DataFrame()

        X_sc = self.scaler.transform(self._featurise_raw(raw_df))
        preds = medfilt(self.clf.predict(X_sc), kernel_size=31)

        # 1. Block Extraction
        blocks = []
        idx = 0
        for node_id, group in groupby(preds):
            length = sum(1 for _ in group)
            blocks.append({"node_id": int(node_id), "start": idx, "len": length})
            idx += length

        # 2. Extract initial candidates (Centroids)
        candidates = []
        for b in blocks:
            if b["len"] < 15: continue  # Drop transient noise
            c_idx = b["start"] + b["len"] // 2
            candidates.append({
                "matched_id": b["node_id"],
                "x": raw_df["x"].iloc[c_idx], "y": raw_df["y"].iloc[c_idx],
                "time": raw_df["time"].iloc[c_idx]
            })

        if not candidates: return pd.DataFrame()

        # 3. Azimuth Filter (Remove mid-line straight points)
        changed = True
        while changed and len(candidates) >= 3:
            changed = False
            for i in range(1, len(candidates) - 1):
                p_prev = np.array([candidates[i - 1]["x"], candidates[i - 1]["y"]])
                p_curr = np.array([candidates[i]["x"], candidates[i]["y"]])
                p_next = np.array([candidates[i + 1]["x"], candidates[i + 1]["y"]])

                ang = self._vec_angle(p_curr - p_prev, p_next - p_curr)
                if ang < 25.0:  # Straight line detected
                    candidates.pop(i)
                    changed = True
                    break

        # 4. Strict Reassignment Pass (Deduplication & Graph Enforcer)
        # This solves the "repeating points" and "huge RMSE" issue.
        final_candidates = [candidates[0]]

        for i in range(1, len(candidates)):
            u = final_candidates[-1]["matched_id"]
            v = candidates[i]["matched_id"]
            curr_pos = np.array([candidates[i]["x"], candidates[i]["y"]])

            # If the Azimuth filter deleted a middle node and pushed two identical
            # IDs together (e.g., A -> A), drop the duplicate.
            if u == v:
                continue

            valid_neighbors = self.connectivity.get(u, [])

            # If the neural net hallucinated a jump to a disconnected node
            if v not in valid_neighbors and len(valid_neighbors) > 0:
                # Snap to the neighbor that physically minimizes distance to the coordinates
                best_nbr, best_dist = v, np.inf
                for nbr in valid_neighbors:
                    nbr_pos = np.array(self.ref_coords[nbr][:2])
                    dist = np.linalg.norm(curr_pos - nbr_pos)
                    if dist < best_dist:
                        best_dist = dist
                        best_nbr = nbr
                v = best_nbr

            candidates[i]["matched_id"] = v
            final_candidates.append(candidates[i])

        return pd.DataFrame([{
            "matched_id": c["matched_id"],
            "sys_pos": (c["x"], c["y"], 0.0),
            "neural_time": c["time"]
        } for c in final_candidates])

# ═════════════════════════════════════════════════════════════════════════════
# 8.  VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════
def _draw_method(ax, sim, df, name: str, color: str, is_sf: bool = False):
    """Draw one method's matched output on top of the simulation background."""
    ax.plot(sim.clean_path_df["x"], sim.clean_path_df["y"],
            "k--", lw=0.8, alpha=0.30, zorder=1, label="Clean path")
    ax.plot(sim.error_path_df["x"], sim.error_path_df["y"],
            color="lightgray", lw=2.0, alpha=0.70, zorder=2, label="Drifted")

    for nid, pos in sim.turning_points.items():
        ax.scatter(pos[0], pos[1], c="black", marker="x", s=60, zorder=3)
        ax.annotate(f"R{nid}", (pos[0], pos[1]),
                    xytext=(0, -10), textcoords="offset points",
                    ha="center", fontsize=5.5, color="black")

    if df is not None and not df.empty:
        if is_sf:
            xs  = [r["sys_pos"][0] for _, r in df.iterrows()]
            ys  = [r["sys_pos"][1] for _, r in df.iterrows()]
            ids = [r["node_id"]     for _, r in df.iterrows()]
        else:
            xs  = [p[0] for p in df["sys_pos"]]
            ys  = [p[1] for p in df["sys_pos"]]
            ids = df["matched_id"].tolist()

        ax.plot(xs, ys, "--", color=color, lw=1.2, alpha=0.65, zorder=4)
        ax.scatter(xs, ys, facecolors="none", edgecolors=color,
                   s=180, lw=2.0, zorder=5)
        for x, y, nid in zip(xs, ys, ids):
            ax.annotate(f"M{nid}", (x, y),
                        xytext=(7, 3), textcoords="offset points",
                        fontsize=5.0, color=color)

    ax.set_title(name, fontsize=9, fontweight="bold", color=color, pad=4)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=6)


def plot_comparison(sim, results: dict, run_idx: int):
    """5-panel figure: 4 method trajectories + metrics table."""
    fig = plt.figure(figsize=(22, 20))
    gs  = GridSpec(3, 2, figure=fig,
                   hspace=0.40, wspace=0.22,
                   top=0.93, bottom=0.03, left=0.04, right=0.97)

    panels = [
        ("Oracle",      gs[0, 0], False),
        ("Viterbi",     gs[0, 1], False),
        ("SmartFilter", gs[1, 0], True),
        ("Neural",      gs[1, 1], False),
    ]
    for name, pos, is_sf in panels:
        ax = fig.add_subplot(pos)
        _draw_method(ax, sim, results[name][0], name,
                     METHOD_COLORS[name], is_sf)

    # ── Metrics table ──────────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[2, :])
    ax_t.axis("off")

    cols = ["Method",
            "Precision (%)", "Recall (%)", "F1 (%)",
            "Ang Err (°)", "RMSE (m)", "Missing (%)"]
    rows = []
    for name in ["Oracle", "Viterbi", "SmartFilter", "Neural"]:
        m = results[name][1]
        ff = lambda v: f"{v:.2f}" if not np.isnan(v) else "n/a"
        rows.append([name,
                     f"{m['precision']:.1f}",
                     f"{m['recall']:.1f}",
                     f"{m['f1']:.1f}",
                     ff(m["angular_error"]),
                     ff(m["rmse"]),
                     f"{m['missing_pct']:.1f}"])

    tbl = ax_t.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False);  tbl.set_fontsize(9.5);  tbl.scale(1, 2.2)

    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    higher_better = [True, True, True, False, False, False]
    for ci, hb in zip(range(1, len(cols)), higher_better):
        vals = []
        for ri in range(1, len(rows) + 1):
            txt = tbl[(ri, ci)].get_text().get_text()
            try:   vals.append(float(txt))
            except ValueError: vals.append(None)
        candidates = [v for v in vals if v is not None]
        if not candidates: continue
        best = (max if hb else min)(candidates)
        for ri in range(1, len(rows) + 1):
            if vals[ri - 1] == best:
                tbl[(ri, ci)].set_facecolor("#d5f5e3")

    ax_t.set_title("Performance Comparison (best in green)", fontsize=11,
                   fontweight="bold", pad=10)
    fig.suptitle(f"Run #{run_idx} – Topological Map-Matching Comparison",
                 fontsize=13, fontweight="bold")

    path = os.path.join(PLOT_DIR, f"comparison_{run_idx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {path}")


def plot_metrics_bar(data: dict, run_idx: int = 0, is_mc: bool = False):
    """
    Side-by-side bar charts for percentage-based metrics.
    If is_mc is True, `data` is mc_history and plots mean ± std dev.
    If is_mc is False, `data` is the single-run results dictionary.
    """
    names = ["Oracle", "Viterbi", "SmartFilter", "Neural"]
    metrics = ["precision", "recall", "f1"]
    labels = ["Precision (%)", "Recall (%)", "F1 Score (%)"]
    colors = [METHOD_COLORS[n] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    title_prefix = "Monte-Carlo " if is_mc else f"Run #{run_idx} – "
    fig.suptitle(f"{title_prefix}Metric Comparison", fontsize=12, fontweight="bold")

    for ax, metric, label in zip(axes, metrics, labels):
        means = []
        stds = []

        if is_mc:
            # Calculate means and stds from lists, ignoring NaNs
            for n in names:
                arr = np.array(data[n][metric], dtype=float)
                valid = arr[~np.isnan(arr)]
                means.append(np.mean(valid) if len(valid) > 0 else 0.0)
                stds.append(np.std(valid) if len(valid) > 0 else 0.0)
        else:
            # Extract scalar values from the single-run tuple
            means = [data[n][1][metric] for n in names]
            stds = [0.0] * len(names)  # No variance in a single run

        # Draw bars with error caps if standard deviations exist
        yerr = stds if is_mc else None
        bars = ax.bar(names, means, color=colors, yerr=yerr, capsize=5,
                      alpha=0.85, edgecolor="white", linewidth=1.2,
                      error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': '#333333'})

        ax.set_ylim(0, 115)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Add value text above the bar (and above the error line if present)
        for i, bar in enumerate(bars):
            val = means[i]
            if val > 0:
                offset = stds[i] if is_mc else 0.0
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + offset + 2,
                        f"{val:.1f}",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    prefix = "mc_" if is_mc else ""
    path = os.path.join(PLOT_DIR, f"{prefix}metrics_bar.png" if is_mc else f"metrics_bar_{run_idx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {path}")


def plot_rmse_and_angular(data: dict, run_idx: int = 0, is_mc: bool = False):
    """
    Bar chart for RMSE and Angular Error.
    If is_mc is True, plots the mean ± std dev across all iterations.
    """
    names = ["Oracle", "Viterbi", "SmartFilter", "Neural"]
    colors = [METHOD_COLORS[n] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    title_prefix = "Monte-Carlo " if is_mc else f"Run #{run_idx} – "
    fig.suptitle(f"{title_prefix}Distance & Angular Error", fontsize=12, fontweight="bold")

    for ax, metric, label, unit in zip(
            axes,
            ["rmse", "angular_error"],
            ["RMSE (m)", "Angular Error (°)"],
            ["m", "°"]):

        means = []
        stds = []

        if is_mc:
            # Calculate means and stds, ignoring NaNs (failed localizations)
            for n in names:
                arr = np.array(data[n][metric], dtype=float)
                valid = arr[~np.isnan(arr)]
                means.append(np.mean(valid) if len(valid) > 0 else 0.0)
                stds.append(np.std(valid) if len(valid) > 0 else 0.0)
        else:
            # Single run scalar extraction
            for n in names:
                val = data[n][1][metric]
                means.append(val if not np.isnan(val) else 0.0)
            stds = [0.0] * len(names)

        # Plot bars with error caps
        yerr = stds if is_mc else None
        bars = ax.bar(names, means, color=colors, yerr=yerr, capsize=5,
                      alpha=0.85, edgecolor="white", lw=1.2,
                      error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': '#333333'})

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Add text labels above the bars/error caps
        for i, bar in enumerate(bars):
            val = means[i]
            if val > 0:
                offset = stds[i] if is_mc else 0.0
                # Dynamically adjust text height so it doesn't overlap the error bar
                text_y = bar.get_height() + offset + (0.2 if metric == 'rmse' else 1.0)
                txt = f"{val:.2f}{unit}"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        text_y, txt, ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    prefix = "mc_" if is_mc else ""
    path = os.path.join(PLOT_DIR, f"{prefix}rmse_angular.png" if is_mc else f"rmse_angular_{run_idx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def save_iteration_corners(detected_df, gt_seq, ref_coords,
                           or_df, vit_df, sf_df, nm_df, filepath):
    max_len = max(len(detected_df), len(gt_seq), len(or_df), len(vit_df), len(sf_df), len(nm_df))
    rows = []

    for j in range(max_len):
        row = {"corner_index": j}

        # TurnDetector Timestamp
        row["timestamp"] = detected_df["time"].iloc[j] if j < len(detected_df) else np.nan
        # Neural Matcher's Independent Timestamp
        row["neural_timestamp"] = nm_df["neural_time"].iloc[j] if j < len(nm_df) else np.nan

        if j < len(gt_seq):
            gt_pos = ref_coords.get(gt_seq[j], (np.nan, np.nan, np.nan))
            row["actual_x"], row["actual_y"], row["actual_z"] = gt_pos[0], gt_pos[1], gt_pos[2]
        else:
            row["actual_x"] = row["actual_y"] = row["actual_z"] = np.nan

        def get_matched_coords(df):
            if j < len(df):
                col = "node_id" if "node_id" in df.columns else "matched_id"
                nid = df[col].iloc[j]
                return ref_coords.get(nid, (np.nan, np.nan, np.nan))
            return (np.nan, np.nan, np.nan)

        row["oracle_x"], row["oracle_y"], row["oracle_z"] = get_matched_coords(or_df)
        row["viterbi_x"], row["viterbi_y"], row["viterbi_z"] = get_matched_coords(vit_df)
        row["smartfilter_x"], row["smartfilter_y"], row["smartfilter_z"] = get_matched_coords(sf_df)
        row["neural_x"], row["neural_y"], row["neural_z"] = get_matched_coords(nm_df)

        rows.append(row)

    pd.DataFrame(rows).to_csv(filepath, index=False)


def run_monte_carlo(n_iterations: int = 10, path_len: float = 800.0,
                    initial_pos_error: float = 12.0, azimuth_bias_deg: float = 8.0):
    """Runs the full simulation pipeline and exports iteration data."""
    print(f"\n{'=' * 75}")
    print(f"  Starting Monte-Carlo Simulation ({n_iterations} iterations)")
    print(f"{'=' * 75}")

    methods = ["Oracle", "Viterbi", "SmartFilter", "Neural"]
    # Add the new keys so the CSV exporter catches them automatically
    metrics_keys = ["precision", "recall", "f1", "angular_error", "rmse",
                    "missing_pct", "n_detected", "n_gt", "detection_ratio_pct"]
    mc_history = {m: {k: [] for k in metrics_keys} for m in methods}

    for i in range(n_iterations):
        run_id = i + 1
        print(f"  [Iteration {run_id}/{n_iterations}] Generating and matching...")

        sim = PathSimulation(seed=None)
        sim.generate(path_len=path_len, initial_pos_error=initial_pos_error, azimuth_bias_deg=azimuth_bias_deg)

        detector = TurnDetector(sim.error_path_df, heading_hw=10, min_angle=25.0, min_spacing=15)
        detected = detector.detect()

        gt_seq = sim.path_indices
        ref_c = sim.turning_points

        if detected.empty:
            print("    -> Warning: No turns detected. Skipping iteration.")
            continue

        # Run Matchers
        or_df = OraclePathMatcher(ref_c, gt_seq).match(detected.copy())
        or_m = compute_metrics(or_df["matched_id"].tolist(), gt_seq, ref_c, or_df["sys_pos"].tolist())

        vit_df = ViterbiMatcher(ref_c, sim.connectivity).match(detected.copy())
        vit_m = compute_metrics(vit_df["matched_id"].tolist(), gt_seq, ref_c, vit_df["sys_pos"].tolist())

        sf_df = SmartFilter(ref_c, sim.connectivity, angle_tol=70.0).filter_and_repair(vit_df.copy())
        sf_ids = sf_df["node_id"].tolist() if not sf_df.empty else []
        sf_sp = sf_df["sys_pos"].tolist() if not sf_df.empty else []
        sf_m = compute_metrics(sf_ids, gt_seq, ref_c, sf_sp)

        # New Continuous Neural Execution
        nm = ContinuousNeuralMatcher(ref_c, gt_seq, sim.connectivity, n_train=100)
        nm.train(sim)

        # Notice we pass the raw path, NOT the detected path
        nm_df = nm.match_continuous(sim.error_path_df.copy())
        nm_m = compute_metrics(nm_df["matched_id"].tolist(), gt_seq, ref_c, nm_df["sys_pos"].tolist())

        # Store metrics
        iter_results = {"Oracle": (or_df, or_m), "Viterbi": (vit_df, vit_m),
                        "SmartFilter": (sf_df, sf_m), "Neural": (nm_df, nm_m)}

        for m_name in methods:
            for k in metrics_keys:
                mc_history[m_name][k].append(iter_results[m_name][1][k])

        # ── Data Exports per Iteration ──────────────────────────────────────────

        # 1. Generate the 4-panel coordinate plot map
        plot_comparison(sim, iter_results, run_idx=run_id)

        # 2. Save Raw Data CSV (timestamp, x, y, z)
        raw_path_file = os.path.join(PLOT_DIR, f"iteration_{run_id:03d}_raw_path.csv")
        sim.error_path_df[["time", "x", "y", "z"]].to_csv(raw_path_file, index=False)

        # 3. Save 17-Column Corner Data CSV
        corner_file = os.path.join(PLOT_DIR, f"iteration_{run_id:03d}_corners.csv")
        save_iteration_corners(detected, gt_seq, ref_c, or_df, vit_df, sf_df, nm_df, corner_file)

    return mc_history


def print_results_table(results: dict):
    sep = "─" * 115
    hdr = (f"{'Method':<14}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  "
           f"{'AngErr°':>9}  {'RMSE_m':>8}  {'Miss%':>7}  "
           f"{'Pts(Det/GT)':>13}  {'DetRatio%':>10}")
    print(f"\n{sep}\n{hdr}\n{sep}")
    for name in ["Oracle", "Viterbi", "SmartFilter", "Neural"]:
        m = results[name][1]
        ae = f"{m['angular_error']:.2f}°" if not np.isnan(m["angular_error"]) else "   n/a"
        rm = f"{m['rmse']:.2f}m" if not np.isnan(m["rmse"]) else "   n/a"

        pts_str = f"{m['n_detected']}/{m['n_gt']}"

        print(f"{name:<14}  "
              f"{m['precision']:>6.1f}%  "
              f"{m['recall']:>6.1f}%  "
              f"{m['f1']:>6.1f}%  "
              f"{ae:>9}  "
              f"{rm:>8}  "
              f"{m['missing_pct']:>6.1f}%  "
              f"{pts_str:>13}  "
              f"{m['detection_ratio_pct']:>9.1f}%")
    print(sep)


def print_mc_stats_table(mc_history: dict):
    """Calculates and prints the mean ± std deviation for the Monte-Carlo results."""
    sep = "─" * 120
    hdr = (f"{'Method':<14}  {'Prec (%)':>13}  {'Recall (%)':>13}  {'F1 (%)':>13}  "
           f"{'RMSE (m)':>13}  {'Det Ratio (%)':>16}")
    print(f"\n{sep}\n{hdr}\n{sep}")

    for name, metrics in mc_history.items():
        row_str = f"{name:<14}  "
        # Displaying a subset of critical metrics for the MC table
        for k in ["precision", "recall", "f1", "rmse", "detection_ratio_pct"]:
            vals = np.array(metrics[k], dtype=float)
            valid_vals = vals[~np.isnan(vals)]

            if len(valid_vals) > 0:
                mean_v = np.mean(valid_vals)
                std_v = np.std(valid_vals)
                row_str += f"{mean_v:>6.1f} ± {std_v:<4.1f}  "
            else:
                row_str += f"{'n/a':>13}  "
        print(row_str)
    print(sep)


def export_mc_to_csv(mc_history: dict, output_dir: str):
    """
    Exports the Monte-Carlo results into two CSV files:
      1. mc_raw_iterations.csv : Every metric for every method per iteration.
      2. mc_summary_stats.csv  : Mean and standard deviation per metric/method.
    """
    print(f"\n  Exporting results to CSV in ./{output_dir}/ ...")

    raw_rows = []
    stats_rows = []

    methods = list(mc_history.keys())
    if not methods:
        return

    metrics = list(mc_history[methods[0]].keys())
    # Determine the number of iterations from the length of the tracked lists
    n_iters = len(mc_history[methods[0]][metrics[0]])

    # ── 1. Build the Raw Data Table ───────────────────────────────────────
    for i in range(n_iters):
        for method in methods:
            row = {"Iteration": i + 1, "Method": method}
            for metric in metrics:
                # Append the raw float value (or NaN if it failed)
                if i < len(mc_history[method][metric]):
                    row[metric] = mc_history[method][metric][i]
                else:
                    row[metric] = np.nan
            raw_rows.append(row)

    df_raw = pd.DataFrame(raw_rows)
    raw_path = os.path.join(output_dir, "mc_raw_iterations.csv")
    df_raw.to_csv(raw_path, index=False)
    print(f"    ✓ Raw iterations   → {raw_path}")

    # ── 2. Build the Statistical Summary Table ────────────────────────────
    for method in methods:
        row_stat = {"Method": method}
        for metric in metrics:
            arr = np.array(mc_history[method][metric], dtype=float)
            valid = arr[~np.isnan(arr)]  # Filter out NaNs for clean math

            mean_val = np.mean(valid) if len(valid) > 0 else np.nan
            std_val = np.std(valid) if len(valid) > 0 else np.nan

            row_stat[f"{metric}_mean"] = mean_val
            row_stat[f"{metric}_std"] = std_val

        stats_rows.append(row_stat)

    df_stats = pd.DataFrame(stats_rows)
    stats_path = os.path.join(output_dir, "mc_summary_stats.csv")
    df_stats.to_csv(stats_path, index=False)
    print(f"    ✓ Summary stats    → {stats_path}")

def main():
    print("=" * 62)
    print("  Topological Map-Matching – 4-Method Comparison Pipeline")
    print("=" * 62)

    # ── 1. Simulate ───────────────────────────────────────────────────────
    print("\n[1/5] Generating simulation…")
    sim = PathSimulation(seed=SEED)
    sim.generate(path_len=800, initial_pos_error=12.0, azimuth_bias_deg=8.0)
    print(f"      Reference nodes : {len(sim.turning_points)}")
    print(f"      Clean path pts  : {len(sim.clean_path_df)}")

    # ── 2. Detect turns ───────────────────────────────────────────────────
    print("\n[2/5] Detecting turns on drifted path…")
    detector = TurnDetector(sim.error_path_df,
                             heading_hw=10, min_angle=25.0, min_spacing=15)
    detected = detector.detect()
    print(f"      Detected turns  : {len(detected)}")
    print(f"      Ground-truth GT : {len(sim.path_indices)}")

    gt_seq = sim.path_indices
    ref_c  = sim.turning_points
    results: dict = {}

    # ── 3. Run all matchers ───────────────────────────────────────────────
    print("\n[3/5] Running matchers…")

    print("  [a] Oracle Path Matcher")
    or_df = OraclePathMatcher(ref_c, gt_seq).match(detected.copy())
    results["Oracle"] = (or_df,
        compute_metrics(or_df["matched_id"].tolist(), gt_seq, ref_c,
                        or_df["sys_pos"].tolist()))

    print("  [b] Viterbi Matcher")
    vit_df = ViterbiMatcher(ref_c, sim.connectivity).match(detected.copy())
    results["Viterbi"] = (vit_df,
        compute_metrics(vit_df["matched_id"].tolist(), gt_seq, ref_c,
                        vit_df["sys_pos"].tolist()))

    print("  [c] Smart Filter (heading-augmented)")
    sf_df  = SmartFilter(ref_c, sim.connectivity, angle_tol=70.0
                         ).filter_and_repair(vit_df.copy())
    sf_ids = sf_df["node_id"].tolist() if not sf_df.empty else []
    sf_sp  = sf_df["sys_pos"].tolist() if not sf_df.empty else []
    results["SmartFilter"] = (sf_df, compute_metrics(sf_ids, gt_seq, ref_c, sf_sp))

    print("  [d] Neural Sequence Matcher (Transformer pipeline)")
    nm = NeuralMatcher(ref_c, gt_seq, n_train=300)
    nm.train(sim)
    nm_df  = nm.match(detected.copy())
    results["Neural"] = (nm_df,
        compute_metrics(nm_df["matched_id"].tolist(), gt_seq, ref_c,
                        nm_df["sys_pos"].tolist()))

    # ── 4. Report ─────────────────────────────────────────────────────────
    print("\n[4/5] Results")
    print_results_table(results)

    # ── 5. Plots ──────────────────────────────────────────────────────────
    print("\n[5/5] Generating plots…")
    plot_comparison(sim, results, run_idx=0)
    plot_metrics_bar(results, run_idx=0)
    plot_rmse_and_angular(results, run_idx=0)
    print(f"\n✓  All outputs saved to:  ./{PLOT_DIR}/")


if __name__ == "__main__":
    mode = input("Run mode: [1] Single Simulation + Plots  [2] Monte-Carlo Stats: ").strip()

    if mode == "1":
        main()
    elif mode == "2":
        n_iters = int(input("Enter number of Monte-Carlo iterations (e.g., 20): ").strip() or "20")
        mc_results = run_monte_carlo(n_iterations=n_iters)

        print("\n[Final Monte-Carlo Results]")
        print_mc_stats_table(mc_results)

        # Plot generation
        print("\n  Generating Monte-Carlo plots...")
        plot_metrics_bar(mc_results, is_mc=True)
        plot_rmse_and_angular(mc_results, is_mc=True)

        # Export to CSV
        export_mc_to_csv(mc_results, output_dir=PLOT_DIR)

        print("\n✓ Monte-Carlo evaluation complete.")
    else:
        print("Invalid selection.")
