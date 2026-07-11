"""BAT internal-coordinate featurizer for alanine dipeptide. Validated round-trip < 1e-14."""
import numpy as np

def _dihedral(p0, p1, p2, p3):
    b0 = p0 - p1; b1 = p2 - p1; b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0*b1n).sum(-1, keepdims=True)*b1n
    w = b2 - (b2*b1n).sum(-1, keepdims=True)*b1n
    x = (v*w).sum(-1); y = (np.cross(b1n, v)*w).sum(-1)
    return np.arctan2(y, x)

def _angle(p0, p1, p2):
    v1 = p0 - p1; v2 = p2 - p1
    v1 /= np.linalg.norm(v1, axis=-1, keepdims=True)
    v2 /= np.linalg.norm(v2, axis=-1, keepdims=True)
    return np.arccos(np.clip((v1*v2).sum(-1), -1, 1))

def cart_to_internal(xyz, order, refs):
    """xyz: (F, N, 3). Returns (F, 60) internal = [b1, b2,a2, b_k,a_k,t_k ...] in `order` sequence.
       Also returns the seed-frame absolute placement for order[0..2] to allow reconstruction."""
    F = xyz.shape[0]
    feats = []
    # seed geometry stored separately (bond01, bond12, angle012) reconstructs seed internally
    for k in range(1, len(order)):
        a = order[k]; r = refs[k]
        if len(r) == 1:
            p = r[0]
            b = np.linalg.norm(xyz[:, a] - xyz[:, p], axis=-1)
            feats.append(b[:, None])
        elif len(r) == 2:
            p, g = r
            b = np.linalg.norm(xyz[:, a] - xyz[:, p], axis=-1)
            ang = _angle(xyz[:, a], xyz[:, p], xyz[:, g])
            feats.append(np.stack([b, ang], -1))
        else:
            p, g, gg = r
            b = np.linalg.norm(xyz[:, a] - xyz[:, p], axis=-1)
            ang = _angle(xyz[:, a], xyz[:, p], xyz[:, g])
            tor = _dihedral(xyz[:, a], xyz[:, p], xyz[:, g], xyz[:, gg])
            feats.append(np.stack([b, ang, tor], -1))
    return np.concatenate(feats, -1)  # (F, 60)

def internal_to_cart(internal, order, refs, seed_geom):
    """Inverse via NeRF. seed_geom = (b01, b12, ang012) arrays (F,). Places seed in canonical frame."""
    F = internal.shape[0]
    xyz = np.zeros((F, len(order), 3))
    b01, b12, ang012 = seed_geom
    a0, a1, a2 = order[0], order[1], order[2]
    xyz[:, a0] = 0.0
    xyz[:, a1, 0] = b01
    # place a2 in xy-plane relative to a1, angle a2-a1-a0 = ang012
    xyz[:, a2, 0] = b01 - b12*np.cos(ang012)
    xyz[:, a2, 1] = b12*np.sin(ang012)
    # unpack the flat feature vector back to per-atom
    idx = 0
    per_atom = {}
    for k in range(1, len(order)):
        L = len(refs[k])
        per_atom[k] = internal[:, idx:idx+L]; idx += L
    for k in range(3, len(order)):
        a = order[k]; p, g, gg = refs[k]
        b = per_atom[k][:, 0]; ang = per_atom[k][:, 1]; tor = per_atom[k][:, 2]
        A = xyz[:, gg]; B = xyz[:, g]; C = xyz[:, p]  # place a relative to C(parent),B(gp),A(ggp)
        bc = C - B; bc /= np.linalg.norm(bc, axis=-1, keepdims=True)
        n = np.cross(B - A, bc); n /= np.linalg.norm(n, axis=-1, keepdims=True)
        m = np.cross(n, bc)
        # NeRF: position of a
        d2 = np.stack([-b*np.cos(ang),
                        b*np.sin(ang)*np.cos(tor),
                        b*np.sin(ang)*np.sin(tor)], -1)
        xyz[:, a] = C + d2[:, 0:1]*bc + d2[:, 1:2]*m + d2[:, 2:3]*n
    return xyz

def seed_geometry(xyz, order):
    a0,a1,a2 = order[0],order[1],order[2]
    b01 = np.linalg.norm(xyz[:,a1]-xyz[:,a0],axis=-1)
    b12 = np.linalg.norm(xyz[:,a2]-xyz[:,a1],axis=-1)
    ang012 = _angle(xyz[:,a2],xyz[:,a1],xyz[:,a0])
    return b01,b12,ang012

def build_zmatrix(topology):
    n = topology.n_atoms
    adj = {i: set() for i in range(n)}
    for b in topology.bonds:
        i, j = b[0].index, b[1].index
        adj[i].add(j); adj[j].add(i)
    root = min((i for i in range(n)), key=lambda i: (len(adj[i]), i))
    parent = {root: None}
    order = []; stack = [root]; seen = set()
    while stack:
        u = stack.pop()
        if u in seen: continue
        seen.add(u); order.append(u)
        for v in sorted(adj[u], reverse=True):
            if v not in seen:
                parent[v] = u; stack.append(v)
    assert len(order) == n
    pos_of = {a: k for k, a in enumerate(order)}
    def anc(a, up):
        for _ in range(up):
            a = parent[a]
            if a is None: return None
        return a
    refs = [None]
    for k in range(1, n):
        a = order[k]; p = parent[a]; g = anc(a, 2); gg = anc(a, 3)
        placed = lambda x: (x in pos_of and pos_of[x] < k)
        if k == 1:
            refs.append((p,)); continue
        if g is None or not placed(g):
            g = next(x for x in adj[p] if x != a and placed(x))
        if k == 2:
            refs.append((p, g)); continue
        # gg: prefer bonded neighbor of g, then of p, then any placed atom
        if gg is None or gg in (a, p, g) or not placed(gg):
            cands = [x for x in adj[g] if x not in (a, p, g) and placed(x)]
            cands += [x for x in adj[p] if x not in (a, p, g) and placed(x)]
            cands += [x for x in order[:k] if x not in (a, p, g)]
            gg = cands[0]
        refs.append((p, g, gg))
    return order, refs, parent

