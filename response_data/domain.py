"""清点检查区间，不构造支撑矩阵或估计时延。"""
from .io import unique,number,positive

def summarize(v):
    unique(v['people'],'id');tail=0;zero=0;weight=0;clusters=set()
    for r in v['people']:
        weight+=positive(r['weight']);clusters.add(r['grid'])
        if number(r['left'])<0:raise ValueError('时延下界不能负')
        if r['right'] is None:tail+=1
        elif number(r['right'])<r['left']:raise ValueError('检查区间逆序')
        if type(r['exact_zero']) is not bool:raise ValueError('零时刻标记应布尔')
        zero+=r['exact_zero']
    return {'people':len(v['people']),'right_censored':tail,'exact_zero':zero,'total_weight':weight,'grids':len(clusters)}
