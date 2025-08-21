import numpy as np


def load_sdplib(sdplib_path, problem_name):
    """
    Convert sdplib problems in SDPA format(http://euler.nmt.edu/~brian/sdplib/)
    to C, A, b, such that the SDP is:

        min <C, X>
        X
        s.t. A(X) = b            →      <A_i, X> = b_i, i = 1, ..., m
            X ⪰ 0

    """

    def _parse_header_and_b(lines):
        """Parses m, nBlocks, block sizes, and b vector (which may span lines)."""
        line_idx = 0
        m = int(lines[line_idx].strip())
        line_idx += 1
        n_blocks = int(lines[line_idx].strip())
        line_idx += 1
        block_sizes = [int(x) for x in lines[line_idx].strip().split()]
        if len(block_sizes) != n_blocks:
            raise ValueError("Number of block sizes does not match n_blocks")
        line_idx += 1
        b_values = []
        while len(b_values) < m and line_idx < len(lines):
            tokens = lines[line_idx].strip().split()
            if tokens:
                b_values.extend([float(x) for x in tokens])
            line_idx += 1
        if len(b_values) != m:
            raise ValueError("Could not parse b vector of length m from file")
        return m, n_blocks, block_sizes, np.array(b_values, dtype=float), line_idx

    def _build_offsets_and_dim(block_sizes):
        sizes_abs = [abs(s) for s in block_sizes]
        offsets = []
        running = 0
        for s in sizes_abs:
            offsets.append(running)
            running += s
        return offsets, running, sizes_abs

    # Read file, skip blank/comment-only lines
    with open(f"{sdplib_path}/{problem_name}.dat-s", "r") as f:
        raw_lines = [ln for ln in (l.strip() for l in f.readlines()) if ln != ""]

    # Header and b
    m, n_blocks, block_sizes, b, start_idx = _parse_header_and_b(raw_lines)
    offsets, n_total, sizes_abs = _build_offsets_and_dim(block_sizes)

    # Initialize matrices F_k for k = 0..m (k=0 corresponds to F_0)
    F = [np.zeros((n_total, n_total), dtype=float) for _ in range(m + 1)]

    # Parse matrix entries: one tuple per line -> (k, block_id, i, j, val)
    for idx in range(start_idx, len(raw_lines)):
        parts = raw_lines[idx].split()
        if len(parts) < 5:
            continue
        k = int(parts[0])  # 0..m (0 is F_0)
        blk = int(parts[1]) - 1  # 0-based
        i = int(parts[2]) - 1  # 0-based within block
        j = int(parts[3]) - 1
        v = float(parts[4])

        if blk < 0 or blk >= n_blocks:
            raise ValueError("Block index out of range in data lines")

        # Map (i, j) within block to global indices
        base = offsets[blk]
        gi = base + i
        gj = base + j

        # If block is diagonal (negative size), ignore off-diagonal entries
        if block_sizes[blk] < 0:
            if gi != gj:
                continue
            F[k][gi, gj] += v
        else:
            F[k][gi, gj] += v
            if gi != gj:
                F[k][gj, gi] += v

    # Ensure exact symmetry
    for k in range(m + 1):
        F[k] = 0.5 * (F[k] + F[k].T)

    # Map SDPA data (F_0, F_i, b) to primal form min <C, X> s.t. <A_i, X> = b_i, X >= 0
    # SDPA uses F_0 + sum x_i F_i ⪰ 0 with objective b^T x (dual form).
    # The corresponding primal uses C = -F_0 and A_i = F_i.
    C = -F[0]
    A = [F[i] for i in range(1, m + 1)]

    # get the solution from the sdplib
    f_sol = sdplib_solution(problem_name)

    return C, A, b, f_sol


def sdplib_solution(name):
    solutions = {
        "arch0": 5.66517e-01,
        "arch2": 6.71515e-01,
        "arch4": 9.726274e-01,
        "arch8": 7.05698e00,
        "control1": 1.778463e01,
        "control2": 8.300000e00,
        "control3": 1.363327e01,
        "control4": 1.979423e01,
        "control5": 1.68836e01,
        "control6": 3.73044e01,
        "control7": 2.06251e01,
        "control8": 2.0286e01,
        "control9": 1.46754e01,
        "control10": 3.8533e01,
        "control11": 3.1959e01,
        "eqaulG11": 6.291553e02,
        "equalG51": 4.005601e03,
        "gpp100": 4.49435e01,
        "gpp124": 7.3431e00,
        "gpp124": 4.68623e01,
        "gpp124": 1.53014e02,
        "gpp124": 4.1899e02,
        "gpp250": 1.5445e01,
        "gpp250": 8.1869e01,
        "gpp250": 3.035e02,
        "gpp250": 7.473e02,
        "gpp500": 2.53e01,
        "gpp500": 1.5606e02,
        "gpp500": 5.1302e02,
        "gpp500": 1.56702e03,
        "hinf1": 2.0326e00,
        "hinf2": 1.0967e01,
        "hinf3": 5.69e01,
        "hinf4": 2.74764e02,
        "hinf5": 3.63e02,
        "hinf6": 4.490e02,
        "hinf7": 3.91e02,
        "hinf8": 1.16e02,
        "hinf9": 2.3625e02,
        "hinf10": 1.09e02,
        "hinf11": 6.59e01,
        "hinf12": 2e-1,
        "hinf13": 4.6e01,
        "hinf14": 1.30e01,
        "hinf15": 2.5e01,
        "infd1": None,
        "infd2": None,
        "infp1": None,
        "infp2": None,
        "maxG11": 6.291648e02,
        "maxG32": 1.567640e03,
        "maxG51": 4.003809e03,
        "maxG55": 9.999210e03,
        "maxG60": 1.522227e04,
        "mcp100": 2.261574e02,
        "mcp124": 1.419905e02,
        "mcp124": 2.698802e02,
        "mcp124": 4.677501e02,
        "mcp124": 8.644119e02,
        "mcp250": 3.172643e02,
        "mcp250": 5.319301e02,
        "mcp250": 9.811726e02,
        "mcp250": 1.681960e03,
        "mcp500": 5.981485e02,
        "mcp500": 1.070057e03,
        "mcp500": 1.847970e03,
        "mcp500": 3.566738e03,
        "qap5": 4.360e02,
        "qap6": 3.8144e02,
        "qap7": 4.25e02,
        "qap8": 7.57e02,
        "qap9": 1.410e03,
        "qap10": 1.093e01,
        "qpG11": 2.448659e03,
        "qpG51": 1.181000e03,
        "ss30": 2.02395e01,
        "theta1": 2.300000e01,
        "theta2": 3.287917e01,
        "theta3": 4.216698e01,
        "theta4": 5.032122e01,
        "theta5": 5.723231e01,
        "theta6": 6.347709e01,
        "thetaG11": 4.000000e02,
        "thetaG51": 3.49000e02,
        "truss1": 8.999996e00,
        "truss2": 1.233804e02,
        "truss3": 9.109996e00,
        "truss4": 9.009996e00,
        "truss5": 1.326357e02,
        "truss6": 9.01001e02,
        "truss7": 9.00001e02,
        "truss8": 1.331146e02,
    }

    return -solutions[name]
