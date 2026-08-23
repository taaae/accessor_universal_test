import csv, json, math, collections, statistics, re
from pathlib import Path

SCR = Path('/private/tmp/claude-501/-Users-tae-Desktop-research-universal-types/ed0ead7a-e556-40f6-adc0-9a2601870313/scratchpad')
LNS = Path('results/022_lns_strategy_performance/run_20260816T211720Z')
IEEE = Path('results/021_unified_strategy_performance/run_20260815T171642Z/unified_core')
COMMON = {'dot': 16777216, 'gemv': 16384}
DISTS = ('uniform_0_1', 'normal_0_1')

def rows(path, scope):
    idx = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r.get('all_valid', '1') != '1': continue
        if int(r['N']) != COMMON[r['kernel']]: continue
        if scope == 'x1' and not (r['access_method'] == 'scalar' and r['packet_values'] == '1'): continue
        idx[(r['arithmetic_type'], r['format'], r['kernel'], r['strategy_id'])].append(r)
    return idx

def score(rs):
    per = {}
    for x in rs:
        if x['distribution'] in DISTS:
            per[x['distribution']] = min(per.get(x['distribution'], 1e9), float(x['median_ms']))
    return math.exp(statistics.fmean(map(math.log, per.values()))) if len(per) == 2 else None

def best(F, S, compute, fmt, kern):
    out = None
    for k in set(F) | set(S):
        if k[:3] != (compute, fmt, kern): continue
        rs = F[k] if k in F else S[k]
        s = score(rs)
        if s and (out is None or s < out[0]): out = (s, k[3], rs[0])
    return out

def lay_of(r, bits, scope):
    if r['storage_layout'] == 'dense': s = 'dense'
    else:
        ctr = 8 if bits <= 8 else 16 if bits <= 16 else 32
        s = 'padded (exact)' if ctr == bits else f'padded({ctr})'
    if scope == 'best' and r['packet_values'] != '1':
        s += f" x{r['packet_values']}"
    return s

# IEEE (E,M) -> node id, from the unified run's format names
EM = {}
for r in csv.DictReader(open(IEEE / 'full/timing_summary.csv')):
    EM[(int(r['exponent_bits']), int(r['mantissa_bits']))] = r['format']

G8 = json.loads((SCR / 'graph8.json').read_text())
LF = {s: rows(LNS / f'{s}/timing_summary.csv', sc) for sc in ('x1', 'best') for s in ('full',) for _ in (0,)}  # placeholder
CACHE = {}
def lns_stage(scope):
    if scope not in CACHE:
        CACHE[scope] = (rows(LNS / 'full/timing_summary.csv', scope),
                        rows(LNS / 'screen/timing_summary.csv', scope))
    return CACHE[scope]

FORMATS = sorted({r['format'] for r in csv.DictReader(open(LNS / 'full/timing_summary.csv'))},
                 key=lambda f: (int(re.match(r'lns(\d+)_r(\d+)', f).group(1)),
                                int(re.match(r'lns(\d+)_r(\d+)', f).group(2))))
COL, TOP, STEP, W = 230, 34, 52, 152

def verdict(x):
    return 'grey' if 0.97 <= x <= 1.03 else ('green' if x > 1.03 else 'red')

OUT = {}
for scope in ('x1', 'best'):
    F, S = lns_stage(scope)
    for compute in ('fp32', 'fp64'):
        for kern in ('dot', 'gemv'):
            g8 = G8[f'{scope}|{compute}|{kern}']
            by = {n['id']: n for n in g8['nodes']}
            nodes, i, missing = [], 0, []
            for f in FORMATS:
                B, R = map(int, re.match(r'lns(\d+)_r(\d+)', f).groups())
                sib_fmt = EM.get((B - 1 - R, R))
                if sib_fmt is None or sib_fmt not in by:
                    missing.append(f); continue
                b = best(F, S, compute, f, kern)
                if b is None:
                    missing.append(f); continue
                ms, sid, row = b
                sib = by[sib_fmt]
                par = by[sib['parent']] if sib['parent'] else None
                y = TOP + STEP * i; i += 1
                nodes.append(dict(id=f'L:{f}', label=f'LNS<{B},{R}>', bits=B, x=2 * COL, y=y,
                                  parent=f'I:{sib_fmt}', ratio=sib['ms'] / ms, ms=ms,
                                  vsraw=g8['rawms'] / ms, strat=sid.split('/')[-1],
                                  lay=lay_of(row, B, scope), peer=sib['label'],
                                  peerstrat=sib['strat'], verdict=verdict(sib['ms'] / ms), kind='lns'))
                nodes.append(dict(id=f'I:{sib_fmt}', label=sib['label'], bits=sib['bits'], x=COL, y=y,
                                  parent=f'P:{par["id"]}' if par else None, ratio=sib['ratio'],
                                  ms=sib['ms'], vsraw=sib['vsraw'], strat=sib['strat'], lay=sib['lay'],
                                  peer=sib['peer'], peerstrat=par['strat'] if par else None,
                                  verdict=sib['verdict'], kind='ieee'))
                if par:
                    gp = by[par['parent']] if par['parent'] else None
                    nodes.append(dict(id=f'P:{par["id"]}', label=par['label'], bits=par['bits'], x=0, y=y,
                                      parent=None, ratio=par['ratio'], ms=par['ms'], vsraw=par['vsraw'],
                                      strat=par['strat'], lay=par['lay'], peer=par['peer'],
                                      peerstrat=gp['strat'] if gp else None,
                                      verdict=par['verdict'], kind='peer'))
            # dedupe shared middle/left boxes, centre each on the chains that reach it
            for kind in ('ieee', 'peer'):
                kids = collections.defaultdict(list)
                for n in nodes:
                    if n['parent']: kids[n['parent']].append(n)
                seen, keep = set(), []
                for n in nodes:
                    if n['kind'] != kind: keep.append(n); continue
                    if n['id'] in seen: continue
                    seen.add(n['id']); keep.append(n)
                    ys = [c['y'] for c in kids[n['id']]]
                    if ys: n['y'] = round(sum(ys) / len(ys))
                nodes = keep
            key = f'lns|{scope}|{compute}|{kern}'
            OUT[key] = dict(nodes=nodes, width=2 * COL + W + 30,
                            height=TOP + STEP * i + 16, rawms=g8['rawms'],
                            rawlabel=g8['rawlabel'], chain=True)
            c = collections.Counter(n['kind'] for n in nodes)
            print(f'{key:24} {c["lns"]:>2} chains  {c["ieee"]:>2} siblings  {c["peer"]:>2} peers'
                  + (f'  skipped {missing}' if missing else ''))

(SCR / 'lns8.json').write_text(json.dumps(OUT))
