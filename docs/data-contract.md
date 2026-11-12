# 收到时延的离散化

support按有限时刻递增，最后一个null代表infinity，且只允许这一个尾原子。普通记录允许L<t<=R，R=null包括所有大于L的有限支撑及infinity；exact_zero仅允许支撑0，left=right=0。普通left=0不包含0，不能与exact_zero混同。

似然是加权记录区间概率之积。CDF在有限t只累计有限支撑<=t；分位为累计概率首次达到alpha的最小支撑，若仅尾原子达到则为infinity。近最优集合loglik>=最佳loglik-drop，drop=0只在声明数值容差内表达最优集合；边界解允许零概率。百分位抽样区间用type-7，出现无限界时保持扩展实数语义，不删除副本。

domain只清点检查区间和整群，不映射支撑或求似然。
