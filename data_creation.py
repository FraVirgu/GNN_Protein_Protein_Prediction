# data_creation.py
# Build a binary DDI dataset from TDC DrugBank:
# - Load all positives (multi-class) and canonicalize pairs
# - Generate negatives (Y=0) by closed-world sampling (not in positives)
# - Shuffle globally, then split into train/valid/test (stratified)
# - Build RDKit graph tensors
# - Reproducible via RANDOM_STATE

from __future__ import annotations
from tdc.multi_pred import DDI
import pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from typing import Dict, Tuple, Optional, List
from rdkit.Chem import rdFMCS


# --------------------------
# Config
# --------------------------
MAX_NUM_NODES = 50
RANDOM_STATE  = 42

# --------------------------
# Helpers: columns & pairs
# --------------------------
def _infer_drug_cols(df: pd.DataFrame) -> Tuple[str, str]:
    for a, b in [("Drug1","Drug2"), ("drug1","drug2")]:
        if a in df.columns and b in df.columns:
            return a, b
    raise ValueError(f"Cannot find Drug1/Drug2 columns in {df.columns.tolist()}")

def _canonical_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)

def _dedupe_unordered_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate unordered pairs so (A,B) and (B,A) won’t appear twice."""
    d1, d2 = _infer_drug_cols(df)
    canon = df[[d1, d2]].apply(lambda r: _canonical_pair(r[d1], r[d2]), axis=1)
    df = df.copy()
    df["_p1"] = [p[0] for p in canon]
    df["_p2"] = [p[1] for p in canon]
    df = df.drop_duplicates(["_p1", "_p2"]).drop(columns=[d1, d2]).rename(columns={"_p1": d1, "_p2": d2})
    return df[[d1, d2] + [c for c in df.columns if c not in (d1, d2)]]

# --------------------------
# Load & merge positives
# --------------------------
def load_all_positives() -> pd.DataFrame:
    """
    Load TDC DrugBank split, merge all splits into a single positives table,
    and deduplicate unordered pairs. Keeps only (Drug1, Drug2).
    """
    data = DDI(name="DrugBank")
    split = data.get_split()  # each split has ['Drug1','Drug2','Y'] where Y is multi-class
    d1, d2 = _infer_drug_cols(split["train"])
    df_all = pd.concat([split["train"][[d1, d2]],
                        split["valid"][[d1, d2]],
                        split["test"] [[d1, d2]]],
                       ignore_index=True)
    df_all = _dedupe_unordered_pairs(df_all)
    return df_all  # positives only

# --------------------------
# Negative sampling
# --------------------------
def _degree_weights_from_pos(df_pos: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    d1, d2 = _infer_drug_cols(df_pos)
    deg = Counter(df_pos[d1]) + Counter(df_pos[d2])
    drugs = list(deg.keys())
    w = np.array([deg[d] for d in drugs], dtype=float)
    w /= w.sum()
    return drugs, w

def sample_negatives_closed_world(
    df_pos: pd.DataFrame,
    n_negatives: int,
    degree_match: bool = True,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Sample Y=0 pairs from all drugs present among positives, excluding any positive pair.
    Treat pairs as unordered. Returns DataFrame with Drug1, Drug2, Y=0.
    """
    rng = np.random.default_rng(random_state)
    d1, d2 = _infer_drug_cols(df_pos)

    # Global positive set for exclusion
    pos_set = set(_canonical_pair(a, b) for a, b in zip(df_pos[d1].values, df_pos[d2].values))

    # Drug pool
    drugs = pd.unique(pd.concat([df_pos[d1], df_pos[d2]], ignore_index=True))
    drugs_list = drugs.tolist()
    n_drugs = len(drugs_list)
    if n_drugs < 2:
        raise ValueError("Not enough unique drugs to sample negatives.")

    # Optional degree-based weights (harder negatives)
    if degree_match:
        pool, w = _degree_weights_from_pos(df_pos)
        wmap = {drug: weight for drug, weight in zip(pool, w)}
        weights = np.array([wmap.get(d, 1e-12) for d in drugs_list], dtype=float)
        weights /= weights.sum()
    else:
        weights = None

    # Sample
    neg_pairs = set()
    max_trials = max(20 * n_negatives, 1_000_000)
    trials = 0
    while len(neg_pairs) < n_negatives and trials < max_trials:
        i = (rng.choice(n_drugs, p=weights) if weights is not None else rng.integers(0, n_drugs))
        j = (rng.choice(n_drugs, p=weights) if weights is not None else rng.integers(0, n_drugs))
        trials += 1
        if i == j: 
            continue
        p = _canonical_pair(drugs_list[i], drugs_list[j])
        if p in pos_set or p in neg_pairs:
            continue
        neg_pairs.add(p)

    if len(neg_pairs) < n_negatives:
        print(f"[warn] requested {n_negatives} negatives, created {len(neg_pairs)}; graph might be dense.")

    neg_df = pd.DataFrame([{"Drug1": a, "Drug2": b, "Y": 0} for (a, b) in neg_pairs])
    return neg_df

# --------------------------
# Build full binary dataframe, shuffle, split
# --------------------------
def make_binary_dataset(
    neg_pos_ratio: Optional[float] = 1.0,    # e.g., 1.0 -> 1:1 neg:pos
    n_negatives: Optional[int] = None,       # exact count overrides ratio
    degree_match: bool = True,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Returns a single shuffled DataFrame with columns ['Drug1','Drug2','Y'] (Y in {0,1}).
    Positives -> Y=1 (all deduped DrugBank pairs). Negatives sampled globally.
    """
    rng = np.random.default_rng(random_state)

    df_pos = load_all_positives()
    df_pos = df_pos[["Drug1", "Drug2"]].copy()
    df_pos["Y"] = 1

    n_pos = len(df_pos)
    if n_negatives is None:
        ratio = 1.0 if neg_pos_ratio is None else float(neg_pos_ratio)
        n_neg = int(np.ceil(ratio * n_pos))
    else:
        n_neg = int(n_negatives)

    df_neg = sample_negatives_closed_world(df_pos, n_neg, degree_match=degree_match, random_state=random_state)

    df_bin = pd.concat([df_pos, df_neg], ignore_index=True)
    # global shuffle (your requirement)
    df_bin = df_bin.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    # small sanity
    vc = df_bin["Y"].value_counts().to_dict()
    print(f"Global label counts (after mix): {vc} (total={len(df_bin)})")
    return df_bin

def stratified_split(
    df_bin: pd.DataFrame,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio:  float = 0.1,
    random_state: int = RANDOM_STATE,
) -> Dict[str, pd.DataFrame]:
    """
    Stratified split by Y to preserve class balance. Ratios must sum to 1.0
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-8, "Ratios must sum to 1."

    rng = np.random.default_rng(random_state)
    out = {}

    parts = []
    for y_val, grp in df_bin.groupby("Y"):
        n = len(grp)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_train = int(round(train_ratio * n))
        n_valid = int(round(valid_ratio * n))
        n_test  = n - n_train - n_valid

        idx_train = idx[:n_train]
        idx_valid = idx[n_train:n_train+n_valid]
        idx_test  = idx[n_train+n_valid:]

        parts.append(("train", grp.iloc[idx_train]))
        parts.append(("valid", grp.iloc[idx_valid]))
        parts.append(("test",  grp.iloc[idx_test]))

    # concat per split and shuffle each split
    for k in ["train", "valid", "test"]:
        df_k = pd.concat([p for name, p in parts if name == k], ignore_index=True)
        df_k = df_k.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        out[k] = df_k

    for k in out:
        print(f"{k} label counts: {out[k]['Y'].value_counts().to_dict()}, n={len(out[k])}")
    return out

# --------------------------
# RDKit graph builders
# --------------------------

from rdkit import Chem
'''
a.GetAtomicNum(),                # 0
a.GetDegree(),                   # 1
a.GetTotalNumHs(),               # 2
int(a.GetIsAromatic()),          # 3
a.GetFormalCharge(),             # 4
a.GetTotalValence(),             # 5
a.GetExplicitValence(),          # 6
a.GetImplicitValence(),          # 7
a.GetNumRadicalElectrons(),      # 8
int(a.IsInRing()),               # 9
int(a.IsInRingSize(3)),          # 10
int(a.IsInRingSize(4)),          # 11
int(a.IsInRingSize(5)),          # 12
int(a.IsInRingSize(6)),          # 13
int(a.GetHybridization()),       # 14 (encode enum as int)
a.GetChiralTag(),                # 15 (also enum → int)
a.GetMass() * 0.01,              # 16 (scaled mass for stability)
'''
def atom_features(a: Chem.Atom) -> list:
    return [
        a.GetAtomicNum(),                # 0
        a.GetDegree(),                   # 1
        a.GetTotalNumHs(),               # 2
        int(a.GetIsAromatic()),          # 3
        a.GetFormalCharge(),             # 4
        a.GetTotalValence(),             # 5
        a.GetExplicitValence(),          # 6
        a.GetImplicitValence(),          # 7
        a.GetNumRadicalElectrons(),      # 8
        int(a.IsInRing()),               # 9
        int(a.IsInRingSize(3)),          # 10
        int(a.IsInRingSize(4)),          # 11
        int(a.IsInRingSize(5)),          # 12
        int(a.IsInRingSize(6)),          # 13
        int(a.GetHybridization()),       # 14 (encode enum as int)
        a.GetChiralTag(),                # 15 (also enum → int)
        a.GetMass() * 0.01              # 16 (scaled mass for stability)
    ]

from rdkit.Chem import rdFMCS

def _safe_find_mcs(mol1, mol2, min_atoms=2, ring_strict=True, timeout=10):
    """
    Call rdFMCS.FindMCS in a way that works across RDKit versions.
    Returns an RDKit MCS result or None.
    """
    opts = rdFMCS.MCSParameters()
    opts.AtomCompare = rdFMCS.AtomCompare.CompareElements
    opts.BondCompare = rdFMCS.BondCompare.CompareOrder
    opts.RingMatchesRingOnly = ring_strict
    opts.CompleteRingsOnly = ring_strict
    opts.Timeout = timeout

    res = None
    try:
        # Most RDKit builds (2022+) use 'parameters='
        res = rdFMCS.FindMCS([mol1, mol2], parameters=opts)
    except TypeError:
        try:
            # Some older examples use 'params='
            res = rdFMCS.FindMCS([mol1, mol2], params=opts)
        except Exception:
            # Last resort: pass kwargs explicitly
            res = rdFMCS.FindMCS(
                [mol1, mol2],
                maximizeBonds=True,
                threshold=1.0,
                timeout=timeout,
                ringMatchesRingOnly=ring_strict,
                completeRingsOnly=ring_strict,
                matchValences=False,
                matchChiralTag=False,
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                bondCompare=rdFMCS.BondCompare.CompareOrder,
            )
    if res is None or res.canceled or res.numAtoms < min_atoms or not res.smartsString:
        return None
    return res

def find_mcs_atom_indices(smi1: str, smi2: str, min_atoms: int = 2, ring_strict: bool = True):
    """
    Returns (idxs1, idxs2) for atoms participating in the MCS, or (None, None).
    Never raises (so it’s safe inside multiprocessing workers).
    """
    try:
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None:
            return None, None

        res = _safe_find_mcs(mol1, mol2, min_atoms=min_atoms, ring_strict=ring_strict)
        if res is None:
            return None, None

        mcs = Chem.MolFromSmarts(res.smartsString)
        if mcs is None:
            return None, None

        m1 = mol1.GetSubstructMatch(mcs)
        m2 = mol2.GetSubstructMatch(mcs)
        if not m1 or not m2:
            return None, None

        return tuple(m1), tuple(m2)
    except Exception:
        # swallow any Boost.Python / FMCS errors to avoid pickling problems
        return None, None

def masked_mcs_adjacency_fullsize(smiles: str, atom_indices, max_nodes: int = MAX_NUM_NODES):
    """
    Full-size adjacency: zeros everywhere except edges among MCS atoms,
    kept at their ORIGINAL indices, then padded to (max_nodes, max_nodes).
    Never raises; returns None only if molecule too large.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros((max_nodes, max_nodes), dtype=np.float32)

        full_adj = rdmolops.GetAdjacencyMatrix(mol).astype(np.float32)  # (n, n)
        n = full_adj.shape[0]
        if n > max_nodes:
            return None

        mask = np.zeros((n,), dtype=np.float32)
        if atom_indices is not None and len(atom_indices) > 0:
            mask[np.asarray(atom_indices, dtype=int)] = 1.0

        keep = np.outer(mask, mask)                   # (n, n)
        mcs_adj = full_adj * keep                     # zero out non-MCS rows/cols

        out = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        out[:n, :n] = mcs_adj
        return out
    except Exception:
        # Safety fallback
        return np.zeros((max_nodes, max_nodes), dtype=np.float32)


def smiles_to_graph(smiles: str, max_nodes: int = MAX_NUM_NODES):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    adj = rdmolops.GetAdjacencyMatrix(mol).astype(np.float32)
    feats = [atom_features(a) for a in mol.GetAtoms()]
    feats = np.asarray(feats, dtype=np.float32)
    n = adj.shape[0]
    if n > max_nodes:
        return None, None
    adj_pad = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    adj_pad[:n, :n] = adj
    feat_dim = feats.shape[1] if n > 0 else 4
    feat_pad = np.zeros((max_nodes, feat_dim), dtype=np.float32)
    if n > 0:
        feat_pad[:n, :] = feats
    return adj_pad, feat_pad

def process_row(row: pd.Series) -> Optional[list]:
    try:
        s1, s2, y = row["Drug1"], row["Drug2"], int(row["Y"])

        # full graphs
        a_adj, a_feat = smiles_to_graph(s1)
        b_adj, b_feat = smiles_to_graph(s2)
        if a_adj is None or b_adj is None:
            return None

        # canonicalize output pair order
        s1c, s2c = _canonical_pair(s1, s2)

        # MCS indices + full-size masked adjacencies
        idxs1, idxs2 = find_mcs_atom_indices(s1, s2, min_atoms=2, ring_strict=True)

        A1_common_full = masked_mcs_adjacency_fullsize(s1, idxs1, max_nodes=MAX_NUM_NODES)
        A2_common_full = masked_mcs_adjacency_fullsize(s2, idxs2, max_nodes=MAX_NUM_NODES)

        # Fallback if molecule too large (None): keep zeros so shapes are consistent
        if A1_common_full is None:
            return None
        if A2_common_full is None:
            return None

        return [s1c, s2c, a_adj, a_feat, b_adj, b_feat, y, A1_common_full, A2_common_full]
    except Exception:
        # Absolutely no exception leaves the worker
        return None

def build_dataset_from_split(df: pd.DataFrame, max_rows: Optional[int] = None,
                             shuffle: bool = True, n_jobs: Optional[int] = None) -> list:
    need = {"Drug1","Drug2","Y"}
    if not need.issubset(df.columns):
        raise ValueError(f"DataFrame must contain {need}, got {df.columns.tolist()}")
    work = df.copy()
    if shuffle:
        work = work.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    if max_rows is not None:
        work = work.iloc[:max_rows].reset_index(drop=True)

    print(f"🔄 Building graphs for {len(work)} pairs in parallel...")
    dataset, skipped = [], 0
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(process_row, row) for _, row in work.iterrows()]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r is None:
                skipped += 1
            else:
                dataset.append(r)
            if i % 1000 == 0 or i == len(futures):
                print(f"  ...processed {i}/{len(futures)} rows")
    print(f"✅ Done. Built {len(dataset)} samples, skipped {skipped} invalid rows.")
    return dataset

# --------------------------
# Public API
# --------------------------
def get_datasets(max_train: int = 5000,
                 max_valid: int = 2000,
                 max_test:  int = 2000,
                 n_jobs: int = 8,
                 # dataset formation controls:
                 neg_pos_ratio: Optional[float] = 1.0, 
                 n_negatives: Optional[int] = None,
                 degree_match: bool = True,
                 # split controls:
                 train_ratio: float = 0.8,
                 valid_ratio: float = 0.1,
                 test_ratio:  float = 0.1,
                 random_state: int = RANDOM_STATE):
    """
    1) Build full binary df (positives + sampled negatives)
    2) Shuffle globally
    3) Stratified random split into train/valid/test
    4) Build graph datasets for each split

    Returns: (train_list, valid_list, test_list, MAX_NUM_NODES)
    """
    # whole mixed binary df
    df_bin = make_binary_dataset(
        neg_pos_ratio=neg_pos_ratio,
        n_negatives=n_negatives,
        degree_match=degree_match,
        random_state=random_state,
    )

    # random stratified split (your requirement)
    split = stratified_split(
        df_bin,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    # build graph datasets
    train = build_dataset_from_split(split["train"], max_rows=max_train, n_jobs=n_jobs)
    valid = build_dataset_from_split(split["valid"], max_rows=max_valid, n_jobs=n_jobs)
    test  = build_dataset_from_split(split["test"],  max_rows=max_test,  n_jobs=n_jobs)
    train = train + valid
    test = test
    return train, valid, test, MAX_NUM_NODES

# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    tr, va, te, _N = get_datasets(
        n_jobs=8,
        neg_pos_ratio=1.0,       # 1:1 negatives vs positives globally
        n_negatives=None,        # or set an exact number and leave ratio=None
        degree_match=True,
        train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1,
    )
    print(f"Train: {len(tr)} | Valid: {len(va)} | Test: {len(te)}")
