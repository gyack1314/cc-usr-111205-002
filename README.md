# 叫应回执的区间删失时延分布

这是独立的防汛演练资料Python项目。`data/`保存虚构人数、观测、时序或方案，`response_data.io`提供严格数值读取与来源字节SHA-256，`response_data.domain`给出已有原始摘要，`response_data.model`构建离散支撑与允许集合，`response_data.fit`做多初值EM极大似然与单纯形KKT诊断，`response_data.bounds`穷举近最优集合给出CDF/分位范围，`response_data.bootstrap`做网格整群bootstrap，`response_data.cli`输出JSON。字段、量纲与分析口径见[资料约定](docs/data-contract.md)。这些数据用于离线复盘，不连接真实人员、传感器或指挥系统。

Python 3.11及以上即可直接使用，无第三方Python依赖。在项目根目录执行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q response_data tests
python -m response_data.cli inspect data/normal.json
python -m response_data.cli estimate data/normal.json
```

`inspect`显示原始资料概况，不覆盖源文件。`estimate`求解非负且和为1的离散支撑概率，最大化加权区间对数似然，输出似然、停止原因、支撑概率、单纯形KKT残差、每人的允许索引与似然贡献、各`loglik_drop`下的近最优CDF/分位范围，以及网格整群bootstrap区间；加`-o 路径`可写入文件。

口径要点：

- 似然为凹问题，EM在“似然列分组”后的单纯形上运行，多个初值用于暴露而非制造多解；设计列相同的候选点在似然层面不可区分，不包装成唯一识别。
- 收敛判据是单纯形KKT残差，不仅凭参数变化小就宣称收敛；达到迭代上限会如实标注`max_iter`。
- 近最优集合按调用者的`loglik_drop`定义，`drop=0`仅在声明数值容差内表达精确最优集；CDF/分位范围由小支撑顶点穷举得到，不是单点解读。
- 不可识别/近最优范围与bootstrap抽样区间分开报告；尾原子承载概率导致的无限分位以`null`加标志如实给出，不删除抽样副本。
- 同一检查区间拆成两条权重各半的记录不改变估计目标（有测试核对）。

`invalid-`前缀资料为不合法输入，其余变体使用相同字段保留不同业务边界。数值null须按字段语义区别于0，不在读取时自动填补。

Dockerfile固定基础镜像，可用`docker build -t response-data .`及`docker run --rm response-data`读取资料；`docker run --rm response-data python -m unittest discover -s tests -v`执行已有检查。运行所需资料全部在本地，无需登录或实时来源。
