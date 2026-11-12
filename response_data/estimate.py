"""估计流水线编排：构建模型 -> 多初值 MLE -> 近最优范围 -> 整群 bootstrap。

输出区分两类不确定性：
  1. 不可识别/近最优范围（似然层面能分辨到的程度，drop=0 为精确多解）；
  2. 网格整群 bootstrap 的抽样区间。
尾原子导致的无限分位以 null + 标志如实报告。
"""
import math
from .model import build, DROP_ZERO_TOL, FEAS_TOL
from .fit import fit, loglik, expand_group_vector, full_kkt_residual, kkt_residual
from .bounds import near_optimal_ranges
from .bootstrap import bootstrap


def _support_label(model, j):
    t = model['support'][j]
    return None if math.isinf(t) else t


def estimate(v):
    model = build(v)
    result = fit(model)
    best = result['best']
    q = best['q']
    weights = [r['weight'] for r in model['rows']]
    ell = best['loglik']

    # 完整 k 维支撑概率（组内均匀的代表性展开）与完整 KKT 残差
    p = expand_group_vector(model, q)
    full_res, full_deriv = full_kkt_residual(model, p)
    group_res, group_deriv = kkt_residual(
        [[model['B'][i][h] for h in model['active_groups']]
         for i in range(len(weights))],
        weights, [q[h] for h in model['active_groups']])

    # 每人的允许索引与对似然的贡献
    people_out = []
    for i, r in enumerate(model['rows']):
        mass = sum(q[h] for h in range(model['g']) if model['B'][i][h])
        contrib = r['weight'] * math.log(mass)
        people_out.append({
            'id': r['id'], 'grid': r['grid'], 'weight': r['weight'],
            'left': r['left'],
            'right': None if math.isinf(r['right']) else r['right'],
            'exact_zero': r['exact_zero'],
            'allowed_support_indices': r['allowed'],
            'allowed_support_times': [_support_label(model, j)
                                      for j in r['allowed']],
            'fitted_interval_mass': mass,
            'loglik_contribution': contrib,
        })

    support_out = []
    for j, t in enumerate(model['support']):
        support_out.append({
            'index': j,
            'time': None if math.isinf(t) else t,
            'is_infinity_tail': math.isinf(t),
            'probability_representative': p[j],
            'likelihood_group': model['col_group'][j],
            'structurally_zero': model['groups'][model['col_group'][j]]['forced_zero'],
        })

    ranges = near_optimal_ranges(model, q, ell)
    boot = bootstrap(model, result)

    # MLE 点的 CDF 与分位（代表性展开），仅作参考，不替代近最优范围
    cdf_point = []
    for t, cut in zip(model['cdf_times'], model['cdf_cut']):
        cdf_point.append({'time': t, 'cdf': sum(p[:cut + 1])})
    quant_point = []
    for a in model['quantiles']:
        cum, qval = 0.0, None
        for j, x in enumerate(p):
            cum += x
            if cum >= a - 1e-12:
                qval = None if math.isinf(model['support'][j]) else model['support'][j]
                break
        quant_point.append({'alpha': a, 'quantile': qval,
                            'is_infinity': qval is None})

    return {
        'method': 'weighted interval-censored MLE on discrete support (EM, convex likelihood)',
        'support_size': model['k'], 'likelihood_groups': model['g'],
        'n_grids': len(model['grids']),
        'loglikelihood': ell,
        'n_parameters_active': len(model['active_groups']),
        'convergence': {
            'stop_reason': best['stop_reason'],
            'iterations': best['iterations'],
            'solver_tolerance': model['tol'],
            'group_simplex_kkt_residual': group_res,
            'full_simplex_kkt_residual': full_res,
            'criterion': 'KKT residual on simplex (not parameter change)',
        },
        'multi_start': [
            {'start_seed': r['start_seed'], 'loglikelihood': r['loglik'],
             'iterations': r['iterations'], 'stop_reason': r['stop_reason'],
             'kkt_residual': r['kkt_residual'],
             'group_probability': r['q']}
            for r in result['runs']
        ],
        'identifiability': {
            'multiple_optima': result['multiple_optima'],
            'ambiguous_likelihood_groups': result['ambiguous_groups'],
            'distinct_optimal_group_vectors': result['distinct_optimal_group_vectors'],
            'note': ('相同设计列的候选点在似然层面不可区分，组内任意再分配均不改变似然；'
                     '多初值仅用于暴露多解，不把重复候选包装成唯一识别。'
                     if result['multiple_optima'] else
                     '在声明数值容差内未发现多解结构。'),
            'drop_zero_tolerance': DROP_ZERO_TOL,
            'feasibility_tolerance': FEAS_TOL,
        },
        'support_probabilities': support_out,
        'people': people_out,
        'point_estimate': {'cdf': cdf_point, 'quantiles': quant_point},
        'near_optimal_ranges': ranges,
        'bootstrap': boot,
        'assumptions_and_limitations': [
            '观察结束时间与检查过程假设独立于真实通知时延。',
            '若存在有意延迟检查（检查时点依赖时延），本估计有偏，该情形只能列为假设限制，数据无法自行纠正。',
            '概率仅定义在配置给定的离散候选时点上，是离散化分析口径，不是连续时间唯一真分布。',
            'right=null 的尾原子表示观察期结束仍未收到；尾原子承载概率时，高分位为 +infinity，如实报告为 null。',
            '近最优/不可识别范围与 bootstrap 抽样区间是两类不同不确定性，分开报告，不可合并。',
        ],
    }
