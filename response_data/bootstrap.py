"""网格整群 bootstrap：按固定种子抽网格，整群复制后重新拟合。

抽样单位为网格：被重复抽中的网格，其全部住户记录与固定权重整体重复。
每次抽样都重新运行 EM 拟合（不扰动已算出的曲线）。
抽样区间与近最优/不可识别范围分开报告；尾原子导致的无限分位如实保留。
百分位区间使用 type-7 定义，无限界保持扩展实数语义，不删除副本。
"""
import math
from .fit import fit_em, expand_group_vector


class _Lcg:
    """与配置种子绑定的确定性线性同余生成器，保证可复现。"""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def randrange(self, n):
        self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
        return self.state % n


def _type7_quantile(sorted_vals, probs):
    """type-7：h=(n-1)*p，线性插值。

    排序时 None 视为扩展实数 +infinity 且排在最后，不删除任何副本。
    若定位或插值涉及尾原子（None），结果如实为 None（+infinity 不可有限化）。
    """
    n = len(sorted_vals)
    out = []
    for p in probs:
        h = (n - 1) * p
        lo = math.floor(h)
        frac = h - lo
        hi_idx = min(lo + 1, n - 1)
        a, b = sorted_vals[lo], sorted_vals[hi_idx]
        if a is None or b is None:
            out.append(None)
        elif frac == 0.0:
            out.append(a)
        else:
            out.append(a + (b - a) * frac)
    return out


def _cdf_at(p, cut):
    return sum(p[j] for j in range(cut + 1))


def _quantile(p, alpha, support):
    cum = 0.0
    for j, x in enumerate(p):
        cum += x
        if cum >= alpha - 1e-12:
            return None if math.isinf(support[j]) else support[j]
    return None


def bootstrap(model, fit_result):
    draws = model['bootstrap_draws']
    if draws == 0:
        return {'draws': 0, 'note': 'bootstrap_draws=0，未执行抽样区间'}
    grids = model['grids']
    by_grid = {g: [] for g in grids}
    for i, r in enumerate(model['rows']):
        by_grid[r['grid']].append(i)
    rng = _Lcg(model['seed'])
    g = model['g']
    active = model['active_groups']
    weights_all = [r['weight'] for r in model['rows']]

    cdf_samples = [[] for _ in model['cdf_cut']]
    q_samples = [[] for _ in model['quantiles']]
    ll_samples, fail = [], 0

    for _ in range(draws):
        # 抽网格（有放回）；整群复制住户与权重
        idx = []
        w = []
        for _ in range(len(grids)):
            g_pick = grids[rng.randrange(len(grids))]
            for i in by_grid[g_pick]:
                idx.append(i)
                w.append(weights_all[i])
        Bb = [[model['B'][i][h] for h in active] for i in idx]
        sub = fit_em(Bb, w, len(active), model['starts'][0], model['tol'],
                     model['max_iter'])
        qfull = [0.0] * g
        for k, h in enumerate(active):
            qfull[h] = sub['q'][k]
        # 完整概率的代表性展开（组内均匀），用于抽样分布
        p = expand_group_vector(model, qfull)
        if sub['stop_reason'] != 'kkt_residual':
            fail += 1
        ll_samples.append(sub['loglik'])
        for c, cut in enumerate(model['cdf_cut']):
            cdf_samples[c].append(_cdf_at(p, cut))
        for c, a in enumerate(model['quantiles']):
            q_samples[c].append(_quantile(p, a, model['support']))

    probs = (0.025, 0.975)

    def ext_sorted(samp):
        # None=+infinity 排最后，保留全部抽样，不删除尾原子副本
        return sorted(samp, key=lambda x: math.inf if x is None else x)

    cdf_ci = []
    for t, samp in zip(model['cdf_times'], cdf_samples):
        lo, hi = _type7_quantile(ext_sorted(samp), probs)
        cdf_ci.append({'time': t, 'ci95_low': lo, 'ci95_high': hi})
    quant_ci = []
    for a, samp in zip(model['quantiles'], q_samples):
        n_inf = sum(1 for x in samp if x is None)
        # 在含 None 的完整样本上做 type-7；不删除无限副本
        lo, hi = _type7_quantile(ext_sorted(samp), probs)
        quant_ci.append({
            'alpha': a,
            'ci95_low': lo, 'ci95_high': hi,
            'infinite_draws': n_inf,
            'all_infinite': n_inf == draws,
            'note': ('存在无限分位抽样，含尾原子的 type-7 定位/插值不可有限化'
                     if n_inf else None),
        })
    return {
        'draws': draws, 'seed': model['seed'], 'resampled_unit': 'grid',
        'grids_sampled': len(grids),
        'draws_not_kkt_converged': fail,
        'cdf_ci95': cdf_ci, 'quantile_ci95': quant_ci,
        'loglik_min': min(ll_samples), 'loglik_max': max(ll_samples),
    }
