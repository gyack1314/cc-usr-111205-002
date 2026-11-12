"""原始资料摘要与区间删失 MLE 估计的命令行入口。"""
import argparse
import json
from .io import load
from .domain import summarize
from .estimate import estimate


def _json_default(o):
    raise TypeError(f'无法序列化类型 {type(o)}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['inspect', 'estimate'])
    p.add_argument('file')
    p.add_argument('-o', '--output', help='可选：把 JSON 结果写入该路径')
    a = p.parse_args()
    v, digest = load(a.file)
    if a.command == 'inspect':
        payload = {'source_sha256': digest, 'summary': summarize(v)}
    else:
        payload = {'source_sha256': digest, 'estimate': estimate(v)}
    text = json.dumps(payload, ensure_ascii=False, indent=2,
                      allow_nan=False, default=_json_default)
    if a.output:
        with open(a.output, 'w', encoding='utf8') as f:
            f.write(text + '\n')
    else:
        print(text)


if __name__ == '__main__':
    main()
