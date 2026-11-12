"""离散支撑区间删失模型的构建与严格校验。

支撑为递增有限时刻并以唯一尾原子 infinity（JSON null）结尾。
普通记录允许 L<t<=R；exact_zero 只允许支撑 0。
本模块只做校验、允许集合与分组设计矩阵，不做任何估计。
"""
import math
from .io import number, positive, unique

FEAS_TOL = 1e-7      # 近最优集合顶点可行性/数值容差
DROP_ZERO_TOL = 1e-7  # drop=0 时声明最优集合所用的数值容差


def _finite_strict_increasing(seq):
    out = []
    for x in seq:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError('有限支撑必须为有限数值')
        if out and x <= out[-1]:
            raise ValueError('有限支撑必须严格递增')
        out.append(float(x))
    return out


def build(v):
    """校验配置并返回模型构件；非法输入抛 ValueError。"""
    if not isinstance(v, dict):
        raise ValueError('顶层资料必须为对象')
    sup = v.get('support')
    if not isinstance(sup, list) or len(sup) < 2 or sup[-1] is not None:
        raise ValueError('support 必须以唯一 infinity 尾原子(null)结尾')
    finite = _finite_strict_increasing(sup[:-1])
    if finite[0] != 0.0:
        raise ValueError('support 必须包含候选时点 0')
    times = finite + [math.inf]
    k = len(times)
    index = {t: j for j, t in enumerate(finite)}

    people = v.get('people')
    if not isinstance(people, list) or not people:
        raise ValueError('people 必须为非空列表')
    unique(people, 'id')
    rows = []
    grids = set()
    for r in people:
        if not isinstance(r, dict):
            raise ValueError('人员记录必须为对象')
        w = positive(r['weight'])  # 固定正权重，零权重拒绝
        grid = r['grid']
        if not isinstance(grid, str) or not grid:
            raise ValueError('网格标识必须非空字符串')
        grids.add(grid)
        if type(r.get('exact_zero')) is not bool:
            raise ValueError('exact_zero 必须为布尔')
        ez = r['exact_zero']
        L = number(r['left'])
        if L < 0:
            raise ValueError('时延下界不能负')
        R = r['right']
        if ez:
            if L != 0 or r['right'] != 0:
                raise ValueError('exact_zero 记录必须 left=right=0')
            allowed = [0]
        else:
            if R is not None:
                R = number(R)
                if R < L:
                    raise ValueError('检查区间逆序')
            # 开左界 L<t，闭右界 t<=R；R=null 含所有大于 L 的有限支撑及尾原子
            allowed = [j for j, t in enumerate(times)
                       if t > L and (R is None or t <= R)]
        if not allowed:
            raise ValueError(f"记录 {r.get('id')} 的允许候选集合为空")
        rows.append({'id': r['id'], 'grid': grid, 'weight': float(w),
                     'left': float(L),
                     'right': (math.inf if R is None else float(R)),
                     'exact_zero': ez, 'allowed': allowed})

    # n x k 0/1 设计矩阵
    A = [[1 if j in rows[i]['allowed'] else 0 for j in range(k)]
         for i in range(len(rows))]

    # 似然只通过 A 的列起作用：相同列的候选点在似然层面不可区分；
    # 全零列（无人允许的候选）构成结构性零概率组
    groups, col_group, seen = [], [], {}
    for j in range(k):
        col = tuple(A[i][j] for i in range(len(rows)))
        if col not in seen:
            seen[col] = len(groups)
            groups.append({'members': [j], 'is_tail': j == k - 1,
                           'forced_zero': 1 not in col})
        else:
            groups[seen[col]]['members'].append(j)
        col_group.append(seen[col])
    g = len(groups)
    G = [[1 if col_group[j] == h else 0 for h in range(g)] for j in range(k)]
    # 组级 n x g 设计矩阵：同组列做逻辑或（0/1 组内列完全相同）
    B = [[0] * g for _ in range(len(rows))]
    for i in range(len(rows)):
        for j in range(k):
            if A[i][j]:
                B[i][col_group[j]] = 1
    active_groups = [h for h, grp in enumerate(groups) if not grp['forced_zero']]
    if not active_groups:
        raise ValueError('没有任何候选被记录允许，配置不可估计')

    cdf_times = v.get('cdf_times', [])
    if not isinstance(cdf_times, list):
        raise ValueError('cdf_times 必须为列表')
    cdf_cut = []
    for t in cdf_times:
        t = number(t)
        if t not in index:
            raise ValueError('cdf_times 必须取自有限支撑候选')
        cdf_cut.append(max(j for j, x in enumerate(finite) if x <= t))

    qs = v.get('quantiles', [])
    if not isinstance(qs, list) or not qs:
        raise ValueError('quantiles 必须为非空列表')
    for a in qs:
        a = number(a)
        if not 0.0 < a < 1.0:
            raise ValueError('分位水平必须严格位于 (0,1)')
    quantiles = [float(a) for a in qs]

    drops = v.get('loglik_drops', [])
    if not isinstance(drops, list) or not drops:
        raise ValueError('loglik_drops 必须为非空列表')
    for d in drops:
        d = number(d)
        if d < 0:
            raise ValueError('loglik_drop 不能为负')
    drops = [float(d) for d in drops]

    draws = v.get('bootstrap_draws', 0)
    if type(draws) is not int or draws < 0:
        raise ValueError('bootstrap_draws 必须为非负整数')
    seed = v.get('seed')
    if type(seed) is not int:
        raise ValueError('seed 必须为整数')
    solv = v.get('solver', {})
    if not isinstance(solv, dict):
        raise ValueError('solver 必须为对象')
    tol = positive(solv.get('tolerance', 1e-9))
    iters = solv.get('iterations', 5000)
    if type(iters) is not int or iters <= 0:
        raise ValueError('iterations 必须为正整数')
    starts = solv.get('starts', [1])
    if not isinstance(starts, list) or not starts or any(type(s) is not int for s in starts):
        raise ValueError('starts 必须为非空整数列表')

    return {
        'support': times, 'finite_support': finite, 'k': k,
        'rows': rows, 'grids': sorted(grids), 'A': A,
        'groups': groups, 'col_group': col_group, 'G': G, 'B': B, 'g': g,
        'active_groups': active_groups,
        'cdf_times': [float(t) for t in cdf_times], 'cdf_cut': cdf_cut,
        'quantiles': quantiles, 'drops': drops,
        'bootstrap_draws': draws, 'seed': seed,
        'tol': float(tol), 'max_iter': iters, 'starts': list(starts),
    }
