"""近最优集合内 CDF 与分位数的精确范围（小支撑顶点穷举）。

近最优集合定义为组空间单纯形上的对数似然凹约束：
    {q>=0, sum q=1, ell(q) >= ell* - drop}
凹函数的水平集为凸多面体；线性目标（CDF）与逐格单调的分位函数，
其上下界都在多面体顶点处取得。多面体的顶点只可能是：
  (a) 单纯形自身顶点（似然约束不活动）；
  (b) 似然水平曲面与单纯形各条边（两自由组、其余钉零）的交点。
凹函数在单变量边上的超水平集是单个区间，先求峰再两侧求根即可穷举。
drop=0 时在声明的数值容差内表达最优集，不把多解包装成唯一识别。
"""
import math
from itertools import combinations
from .model import DROP_ZERO_TOL, FEAS_TOL
from .fit import loglik


def _golden_max(f, lo, hi, it=70):
    phi = (math.sqrt(5.0) - 1) / 2
    x1, x2 = hi - phi * (hi - lo), lo + phi * (hi - lo)
    f1, f2 = f(x1), f(x2)
    for _ in range(it):
        if f1 < f2:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + phi * (hi - lo)
            f2 = f(x2)
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - phi * (hi - lo)
            f1 = f(x1)
    xm = (lo + hi) / 2
    return xm, f(xm)


def _root(f, lo, hi, it=90):
    """求 f=0 的根，要求 f(lo)<=0<=f(hi) 或反向。"""
    flo, fhi = f(lo), f(hi)
    for _ in range(it):
        mid = (lo + hi) / 2
        fm = f(mid)
        if (flo <= 0 <= fm) or (flo >= 0 >= fm):
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


def enumerate_vertices(model, q_star, ell_star, drop):
    """枚举近最优多面体在活跃组空间上的全部顶点（含强制零组补齐）。"""
    B, weights = model['B'], [r['weight'] for r in model['rows']]
    active = model['active_groups']
    ga = len(active)
    Ba = [[B[i][h] for h in active] for i in range(len(weights))]
    threshold = ell_star - drop

    def ell_a(z):
        return loglik(Ba, weights, z)

    def feasible(z):
        if any(x < -FEAS_TOL for x in z):
            return False
        if abs(sum(z) - 1.0) > FEAS_TOL:
            return False
        return ell_a(z) + FEAS_TOL >= threshold

    def lift(z):
        q = [0.0] * model['g']
        for k, h in enumerate(active):
            q[h] = max(0.0, z[k])
        return q

    verts = []

    def add(z):
        if not feasible(z):
            return
        z = [max(0.0, x) for x in z]
        s = sum(z)
        z = [x / s for x in z] if s else z
        for v in verts:
            if max(abs(a - b) for a, b in zip(v, z)) <= 1e-8:
                return
        verts.append(z)

    # (a) 似然约束不活动：单纯形顶点
    for k in range(ga):
        add([1.0 if t == k else 0.0 for t in range(ga)])

    # (b) 似然约束活动：枚举单纯形每条边与水平曲面的交点
    for u, w in combinations(range(ga), 2):
        def h(x, u=u, w=w):
            z = [0.0] * ga
            z[u], z[w] = x, 1.0 - x
            return ell_a(z) - threshold

        xm, hm = _golden_max(h, 0.0, 1.0)
        if hm < -FEAS_TOL:
            continue  # 整条边都不可行
        h0, h1 = h(0.0), h(1.0)
        if h0 < -FEAS_TOL:  # 左端点在域外 -> 左侧根
            x = _root(h, 0.0, xm)
            z = [0.0] * ga
            z[u], z[w] = x, 1.0 - x
            add(z)
        if h1 < -FEAS_TOL:  # 右端点在域外 -> 右侧根
            x = _root(h, xm, 1.0)
            z = [0.0] * ga
            z[u], z[w] = x, 1.0 - x
            add(z)

    # (c) 水平超曲面在单纯形内部形成闭多面体（典型于 drop>0、MLE 为内点）：
    # 从内部最优点 q* 向每个单纯形顶点作射线，与水平曲面的交点即该多面体顶点
    zstar = [q_star[h] for h in active]
    if ell_a(zstar) + FEAS_TOL >= threshold:
        for k in range(ga):
            if zstar[k] > 1.0 - 1e-12:
                continue
            # z(t) = zstar + t*(e_k - zstar)，t in [0,1]，t=1 为单纯形顶点
            def ray(t, k=k):
                z = [(1 - t) * zstar[j] + (t if j == k else 0.0)
                     for j in range(ga)]
                return ell_a(z) - threshold

            r1 = ray(1.0)
            if r1 >= -FEAS_TOL:
                continue  # 该方向到单纯形顶点仍可行，边界点已由 (a)/(b) 覆盖
            t = _root(ray, 0.0, 1.0)
            add([(1 - t) * zstar[j] + (t if j == k else 0.0)
                 for j in range(ga)])

    # drop=0：MLE 自身（可能在边界面或内部），在数值容差内纳入
    add(zstar)
    return [lift(z) for z in verts]


def _group_cdf_limits(model, q, cut):
    """给定组概率 q，CDF(<= cut) 在组内再分配下的最小/最大值。"""
    lo = hi = 0.0
    for h, grp in enumerate(model['groups']):
        members = grp['members']
        n_lo = sum(1 for j in members if j <= cut)
        if n_lo == len(members):
            lo += q[h]
            hi += q[h]
        elif n_lo > 0:
            hi += q[h]  # 全放最早成员（<=cut）；下界放最晚成员，贡献 0
    return lo, hi


def _group_quantile_limits(model, q, alpha):
    """给定组概率 q，分位数在组内再分配下的最小/最大支撑（扩展实数）。"""
    k = model['k']
    p_early = [0.0] * k
    p_late = [0.0] * k
    for h, grp in enumerate(model['groups']):
        p_early[grp['members'][0]] += q[h]
        p_late[grp['members'][-1]] += q[h]

    def quant(p):
        cum = 0.0
        for j in range(k):
            cum += p[j]
            if cum >= alpha - 1e-12:
                return j
        return k - 1

    times = model['support']
    return times[quant(p_early)], times[quant(p_late)]


def near_optimal_ranges(model, q_star, ell_star):
    """对每个 loglik_drop 给出 CDF 与分位数的近最优范围。"""
    out = []
    for drop in model['drops']:
        verts = enumerate_vertices(model, q_star, ell_star, drop)
        if not verts:
            raise RuntimeError(f'drop={drop} 近最优顶点枚举为空')
        cdf = []
        for t, cut in zip(model['cdf_times'], model['cdf_cut']):
            los = [_group_cdf_limits(model, q, cut)[0] for q in verts]
            his = [_group_cdf_limits(model, q, cut)[1] for q in verts]
            cdf.append({'time': t, 'cdf_min': min(los), 'cdf_max': max(his)})
        quant = []
        for a in model['quantiles']:
            los, his = [], []
            for q in verts:
                lo, hi = _group_quantile_limits(model, q, a)
                los.append(lo)
                his.append(hi)
            qmin, qmax = min(los), max(his)
            quant.append({'alpha': a,
                          'q_min': None if math.isinf(qmin) else qmin,
                          'q_max': None if math.isinf(qmax) else qmax,
                          'q_min_is_infinity': math.isinf(qmin),
                          'q_max_is_infinity': math.isinf(qmax)})
        out.append({'loglik_drop': drop,
                    'drop_zero_tolerance': DROP_ZERO_TOL if drop <= DROP_ZERO_TOL else None,
                    'n_vertices': len(verts),
                    'cdf': cdf, 'quantiles': quant})
    return out
