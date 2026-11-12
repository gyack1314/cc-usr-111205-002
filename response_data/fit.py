"""加权区间删失 MLE：分组空间 EM、多初值与单纯形 KKT 诊断。

目标函数（凹但允许多解）：
    ell(p) = sum_i w_i * log( sum_{j 允许于 i} p_j )
EM 在“似然列分组”后的 g 维单纯形上运行；分组内候选点在似然层面不可区分，
任何组内再分配都不改变似然，多解由分组结构与多初值共同揭示。
收敛判据使用单纯形 KKT 残差，不仅凭参数变化小就宣称收敛。
"""
import math
from .model import DROP_ZERO_TOL


def loglik(B, weights, q):
    """组空间对数似然；q 为 g 维概率。"""
    total = 0.0
    for i, w in enumerate(weights):
        m = 0.0
        Bi = B[i]
        for h in range(len(q)):
            if Bi[h]:
                m += q[h]
        if m <= 0.0:
            return math.inf * (-1.0)
        total += w * math.log(m)
    return total


def kkt_residual(B, weights, q):
    """完整单纯形 KKT 残差（组空间）。

    一阶条件：对每个支撑 h，导数 d_h = sum_{i:h 允许} w_i / m_i。
    最优时存在乘子 lambda 使 q_h>0 => d_h=lambda；q_h=0 => d_h<=lambda。
    返回 max(活跃导数极差, 非活跃越界, 单纯形约束违背)。
    """
    g = len(q)
    n = len(weights)
    masses = [0.0] * n
    for i in range(n):
        Bi = B[i]
        m = 0.0
        for h in range(g):
            if Bi[h]:
                m += q[h]
        masses[i] = m
    deriv = [0.0] * g
    for h in range(g):
        s = 0.0
        for i in range(n):
            if B[i][h]:
                s += weights[i] / masses[i]
        deriv[h] = s
    active = [h for h in range(g) if q[h] > 1e-12]
    lam = sum(deriv[h] for h in active) / len(active) if active else 0.0
    res = 0.0
    if active:
        res = max(abs(deriv[h] - lam) for h in active)
    for h in range(g):
        if q[h] <= 1e-12:
            res = max(res, deriv[h] - lam)
            res = max(res, -q[h])
    res = max(res, abs(sum(q) - 1.0))
    return res, deriv


def _em_step(B, weights, q):
    n = len(weights)
    g = len(q)
    nq = [0.0] * g
    total_w = sum(weights)
    for i in range(n):
        Bi = B[i]
        m = 0.0
        for h in range(g):
            if Bi[h]:
                m += q[h]
        wi = weights[i]
        for h in range(g):
            if Bi[h]:
                nq[h] += wi * q[h] / m
    return [x / total_w for x in nq]


def _initial(g, seed):
    """确定性初值：seed=1 为均匀，其余为确定性 Dirichlet 式扰动。"""
    if seed == 1:
        return [1.0 / g] * g
    rng_state = (seed * 2654435761) & 0xFFFFFFFF
    raw = []
    for _ in range(g):
        rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
        # 确定性对数均匀分量，保证严格正
        raw.append(1e-3 + rng_state / 0x7FFFFFFF)
    s = sum(raw)
    return [x / s for x in raw]


def fit_em(B, weights, g, seed, tol, max_iter):
    """单次 EM 拟合，返回概率、似然、迭代信息与停止原因。"""
    q = _initial(g, seed)
    ell_prev = loglik(B, weights, q)
    stop = 'max_iter'
    for it in range(1, max_iter + 1):
        q = _em_step(B, weights, q)
        ell = loglik(B, weights, q)
        if not math.isfinite(ell):
            raise RuntimeError('似然非有限：数据或初值异常')
        res, _ = kkt_residual(B, weights, q)
        # 主判据是 KKT 残差，参数/似然变化仅作辅助记录
        if res <= tol:
            stop = 'kkt_residual'
            ell_prev = ell
            break
        ell_prev = ell
    res, deriv = kkt_residual(B, weights, q)
    return {'q': q, 'loglik': ell_prev, 'iterations': it,
            'stop_reason': stop, 'kkt_residual': res, 'derivatives': deriv,
            'start_seed': seed}


def fit(model):
    """多初值拟合，取似然最高者并保留全部初值结果以暴露多解。"""
    weights = [r['weight'] for r in model['rows']]
    active = model['active_groups']
    Ba = [[model['B'][i][h] for h in active] for i in range(len(weights))]
    ga = len(active)
    sub_runs = [fit_em(Ba, weights, ga, s, model['tol'], model['max_iter'])
                for s in model['starts']]
    # 把活跃子空间结果提升回完整组空间（强制零组保持 0）
    runs = []
    for r in sub_runs:
        qfull = [0.0] * model['g']
        for k, h in enumerate(active):
            qfull[h] = r['q'][k]
        runs.append({**r, 'q': qfull})
    best = max(runs, key=lambda r: r['loglik'])
    ll = best['loglik']
    # 不同初值收敛到似然相同但概率不同的点 => 多解证据
    distinct = []
    for r in runs:
        if abs(r['loglik'] - ll) <= DROP_ZERO_TOL:
            if not any(max(abs(a - b) for a, b in zip(r['q'], q0)) <= 1e-6
                       for q0 in distinct):
                distinct.append(r['q'])
    # 结构不可识别：存在含多于一个候选点的似然列分组
    ambiguous_groups = [h for h, grp in enumerate(model['groups'])
                        if len(grp['members']) > 1]
    return {
        'best': best, 'runs': runs,
        'distinct_optimal_group_vectors': distinct,
        'multiple_optima': len(distinct) > 1 or bool(ambiguous_groups),
        'ambiguous_groups': ambiguous_groups,
    }


def expand_group_vector(model, q):
    """把组空间概率展开为完整 k 维概率：组内均匀分配（仅代表性展示）。"""
    p = [0.0] * model['k']
    for h, grp in enumerate(model['groups']):
        share = q[h] / len(grp['members'])
        for j in grp['members']:
            p[j] = share
    return p


def full_kkt_residual(model, p):
    """完整 k 维概率上的单纯形 KKT 残差（含组内零概率分量诊断）。"""
    A, weights, k = model['A'], [r['weight'] for r in model['rows']], model['k']
    n = len(weights)
    masses = [0.0] * n
    for i in range(n):
        Ai = A[i]
        m = 0.0
        for j in range(k):
            if Ai[j]:
                m += p[j]
        masses[i] = m
    deriv = [0.0] * k
    for j in range(k):
        s = 0.0
        for i in range(n):
            if A[i][j]:
                s += weights[i] / masses[i]
        deriv[j] = s
    active = [j for j in range(k) if p[j] > 1e-12]
    lam = sum(deriv[j] for j in active) / len(active)
    res = max(abs(deriv[j] - lam) for j in active)
    for j in range(k):
        if p[j] <= 1e-12:
            res = max(res, deriv[j] - lam, -p[j])
    return max(res, abs(sum(p) - 1.0)), deriv
