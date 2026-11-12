"""区间删失 MLE、近最优界、不可识别与整群 bootstrap 的检查。"""
import math
import unittest

from response_data.io import load
from response_data.model import build
from response_data.fit import fit, loglik, expand_group_vector, full_kkt_residual
from response_data.bounds import enumerate_vertices, near_optimal_ranges
from response_data.estimate import estimate


def _load(name):
    if not name.endswith('.json'):
        name += '.json'
    return load(f'data/{name}')[0]


class EstimationTests(unittest.TestCase):
    def test_normal_mle_and_kkt(self):
        m = build(_load('normal.json'))
        r = fit(m)
        best = r['best']
        self.assertEqual(best['stop_reason'], 'kkt_residual')
        self.assertLess(best['kkt_residual'], 1e-7)
        # 手工 MLE：p0=1/6, p5=5/9, p10=5/18，p2=p_tail=0（边界解允许零概率）
        q = best['q']
        self.assertAlmostEqual(q[0], 1 / 6, places=8)
        self.assertAlmostEqual(q[2], 5 / 9, places=8)
        self.assertAlmostEqual(q[3], 5 / 18, places=8)
        self.assertLess(q[1], 1e-8)
        self.assertLess(q[4], 1e-8)
        self.assertAlmostEqual(sum(q), 1.0, places=12)
        p = expand_group_vector(m, q)
        res, _ = full_kkt_residual(m, p)
        self.assertLess(res, 1e-7)
        # 完整概率和为 1、非负
        self.assertTrue(all(x >= -1e-12 for x in p))
        self.assertAlmostEqual(sum(p), 1.0, places=12)

    def test_multi_start_agrees_on_identified_case(self):
        m = build(_load('normal.json'))
        r = fit(m)
        lls = [run['loglik'] for run in r['runs']]
        self.assertAlmostEqual(max(lls), min(lls), places=7)
        self.assertFalse(r['multiple_optima'])

    def test_structural_nonidentifiability(self):
        m = build(_load('nonunique-optimal-set.json'))
        r = fit(m)
        self.assertTrue(r['multiple_optima'])
        # 候选 0 与尾原子 4 同为全零列 -> 结构零组；2,5,10 设计列相同 -> 模糊组
        self.assertTrue(m['groups'][0]['forced_zero'])
        self.assertEqual(set(m['groups'][1]['members']), {1, 2, 3})
        # 似然恒为 0（两条记录都允许全部有限非零候选）
        self.assertAlmostEqual(r['best']['loglik'], 0.0, places=10)
        ranges = near_optimal_ranges(m, r['best']['q'], r['best']['loglik'])
        cdf2 = ranges[0]['cdf'][0]
        # drop=0 精确多解：CDF(2) 在 [0,1] 全范围
        self.assertAlmostEqual(cdf2['cdf_min'], 0.0, places=8)
        self.assertAlmostEqual(cdf2['cdf_max'], 1.0, places=8)
        self.assertEqual(ranges[0]['drop_zero_tolerance'], 1e-7)

    def test_tail_only_infinite_quantile(self):
        out = estimate(_load('only-tail-information.json'))
        q09 = out['near_optimal_ranges'][0]['quantiles'][1]
        self.assertTrue(q09['q_max_is_infinity'])
        self.assertIsNone(q09['q_max'])
        boot = out['bootstrap']['quantile_ci95'][1]
        self.assertTrue(boot['all_infinite'])
        self.assertIsNone(boot['ci95_low'])

    def test_near_optimal_contains_mle_and_widens(self):
        m = build(_load('normal.json'))
        r = fit(m)
        ell = r['best']['loglik']
        v0 = enumerate_vertices(m, r['best']['q'], ell, 0.0)
        v5 = enumerate_vertices(m, r['best']['q'], ell, 0.5)
        self.assertGreaterEqual(len(v5), len(v0))
        # drop=0.5 的 CDF 界必须包住 MLE 点且宽度非零
        ranges = near_optimal_ranges(m, r['best']['q'], ell)
        c = ranges[1]['cdf'][1]  # CDF(5)
        self.assertLess(c['cdf_min'], c['cdf_max'])
        p = expand_group_vector(m, r['best']['q'])
        cdf5 = sum(p[:3])
        self.assertLessEqual(c['cdf_min'] - 1e-9, cdf5)
        self.assertLessEqual(cdf5, c['cdf_max'] + 1e-9)

    def test_vertex_feasibility(self):
        for name in ('normal', 'nonunique-optimal-set', 'only-tail-information'):
            m = build(_load(name))
            r = fit(m)
            ell = r['best']['loglik']
            w = [row['weight'] for row in m['rows']]
            for drop in m['drops']:
                for q in enumerate_vertices(m, r['best']['q'], ell, drop):
                    self.assertTrue(all(x >= -1e-7 for x in q))
                    self.assertAlmostEqual(sum(q), 1.0, places=6)
                    self.assertGreaterEqual(loglik(m['B'], w, q) + 1e-7,
                                            ell - drop)

    def test_bootstrap_resamples_grids_and_refits(self):
        out = estimate(_load('normal.json'))
        b = out['bootstrap']
        self.assertEqual(b['draws'], 99)
        self.assertEqual(b['resampled_unit'], 'grid')
        self.assertEqual(b['grids_sampled'], 3)
        self.assertEqual(b['draws_not_kkt_converged'], 0)
        for c in b['cdf_ci95']:
            self.assertLessEqual(c['ci95_low'], c['ci95_high'] + 1e-12)
            self.assertTrue(0.0 <= c['ci95_low'] <= 1.0)
            self.assertTrue(0.0 <= c['ci95_high'] <= 1.0)

    def test_bootstrap_deterministic_with_seed(self):
        v = _load('normal.json')
        b1 = estimate(v)['bootstrap']
        b2 = estimate(v)['bootstrap']
        self.assertEqual(b1['cdf_ci95'], b2['cdf_ci95'])

    def test_split_record_invariance(self):
        base = {
            'support': [0, 2, 5, 10, None],
            'people': [
                {'id': 'a', 'grid': 'g1', 'left': 0, 'right': 5,
                 'exact_zero': False, 'weight': 2},
                {'id': 'b', 'grid': 'g2', 'left': 2, 'right': 10,
                 'exact_zero': False, 'weight': 1},
                {'id': 'c', 'grid': 'g2', 'left': 5, 'right': None,
                 'exact_zero': False, 'weight': 1},
            ],
            'cdf_times': [2, 5, 10], 'quantiles': [0.5, 0.9],
            'loglik_drops': [0, 0.5], 'bootstrap_draws': 11, 'seed': 7,
            'solver': {'tolerance': 1e-9, 'iterations': 5000, 'starts': [1, 7]},
        }
        import copy
        split = copy.deepcopy(base)
        split['people'] = [
            {'id': 'a1', 'grid': 'g1', 'left': 0, 'right': 5,
             'exact_zero': False, 'weight': 1},
            {'id': 'a2', 'grid': 'g1', 'left': 0, 'right': 5,
             'exact_zero': False, 'weight': 1},
            base['people'][1], base['people'][2],
        ]
        e1, e2 = estimate(base), estimate(split)
        self.assertAlmostEqual(e1['loglikelihood'], e2['loglikelihood'],
                               places=10)
        p1 = [s['probability_representative'] for s in e1['support_probabilities']]
        p2 = [s['probability_representative'] for s in e2['support_probabilities']]
        self.assertLess(max(abs(a - b) for a, b in zip(p1, p2)), 1e-7)
        for r1, r2 in zip(e1['near_optimal_ranges'], e2['near_optimal_ranges']):
            for c1, c2 in zip(r1['cdf'], r2['cdf']):
                self.assertAlmostEqual(c1['cdf_min'], c2['cdf_min'], places=8)
                self.assertAlmostEqual(c1['cdf_max'], c2['cdf_max'], places=8)
        ca = next(x for x in e1['people'] if x['id'] == 'a')['loglik_contribution']
        cs = sum(x['loglik_contribution'] for x in e2['people']
                 if x['id'] in ('a1', 'a2'))
        self.assertAlmostEqual(ca, cs, places=10)

    def test_person_allowed_indices_and_contributions(self):
        out = estimate(_load('normal.json'))
        ids = {p['id']: p for p in out['people']}
        self.assertEqual(ids['h1']['allowed_support_indices'], [0])
        self.assertEqual(ids['h4']['allowed_support_indices'], [3, 4])
        # 各人贡献之和等于总似然
        self.assertAlmostEqual(
            sum(p['loglik_contribution'] for p in out['people']),
            out['loglikelihood'], places=10)

    def test_validation_rejects_bad_inputs(self):
        good = _load('normal.json')
        # 空允许集合
        bad = __import__('copy').deepcopy(good)
        bad['people'][0]['exact_zero'] = False
        bad['people'][0]['right'] = -1
        with self.assertRaises(ValueError):
            estimate(bad)
        # 尾原子必须唯一且在末尾
        bad = __import__('copy').deepcopy(good)
        bad['support'] = [0, 2, 5, 10]
        with self.assertRaises(ValueError):
            estimate(bad)
        # 分位越界
        bad = __import__('copy').deepcopy(good)
        bad['quantiles'] = [1.0]
        with self.assertRaises(ValueError):
            estimate(bad)
        # 零权重拒绝
        with self.assertRaises(ValueError):
            estimate(_load('invalid-domain.json'))

    def test_exact_zero_distinct_from_left_zero(self):
        m = build(_load('normal.json'))
        rows = {r['id']: r for r in m['rows']}
        self.assertEqual(rows['h1']['allowed'], [0])          # exact_zero 只含 0
        self.assertNotIn(0, rows['h2']['allowed'])            # 普通 left=0 不含 0


if __name__ == '__main__':
    unittest.main()
