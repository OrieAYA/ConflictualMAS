import csv, statistics as st
from collections import defaultdict

rows = list(csv.DictReader(open('C:/ConflictualMAS/results/episodes_seed42.csv')))
print(f'Total rows: {len(rows)}')

def num(s):
    try: return float(s)
    except: return None

# Structure: per (mode, city) aggregate over rounds
by_mc = defaultdict(list)
for r in rows:
    by_mc[(r['policy_mode'], r['city'])].append(r)

print('\n=== Mode x City counts ===')
for k in sorted(by_mc): print(f'  {k}: {len(by_mc[k])} eps')

# Headline metrics per mode x city
print('\n=== Headline (mean over all 80 rounds) ===')
cols = ['throughput_rate','accept_rate','completion_per_accepted','latency_mean',
        'agent_completed_gini','agent_completed_std',
        'mean_imp_accepted','mean_imp_refused',
        'accept_rate_high_cong','accept_rate_low_cong',
        'mean_congestion_at_decision','n_ghost_active_mean',
        'mean_extra_steps_per_task','wallclock_ms',
        'actor_loss','entropy']
hdr = f"{'Mode':<16}{'City':<22}" + ''.join(f'{c[:14]:>15}' for c in cols)
print(hdr)
for k in sorted(by_mc):
    line = f"{k[0]:<16}{k[1]:<22}"
    for c in cols:
        vals = [num(r[c]) for r in by_mc[k] if num(r[c]) is not None]
        m = st.mean(vals) if vals else 0
        line += f"{m:>15.3f}"
    print(line)

# Convergence: split into early (rounds 0-19) vs late (rounds 60-79)
print('\n=== Convergence (early rounds 0-19 vs late 60-79) ===')
def slice_rounds(rows_mc, lo, hi):
    # global_episode increases over training but rounds are striped across cities/modes.
    # Use the first half / last quarter by record index instead.
    n = len(rows_mc)
    return rows_mc[int(lo/80*n):int(hi/80*n)]

for k in sorted(by_mc):
    early = slice_rounds(by_mc[k], 0, 20)
    late  = slice_rounds(by_mc[k], 60, 80)
    def m(rows, col):
        vs = [num(r[col]) for r in rows if num(r[col]) is not None]
        return st.mean(vs) if vs else 0
    print(f'  {k[0]:<16}{k[1]:<22}'
          f'  thr early={m(early,"throughput_rate"):.3f} late={m(late,"throughput_rate"):.3f}'
          f'  acc early={m(early,"accept_rate"):.3f} late={m(late,"accept_rate"):.3f}'
          f'  c/a early={m(early,"completion_per_accepted"):.3f} late={m(late,"completion_per_accepted"):.3f}'
          f'  aloss early={m(early,"actor_loss"):.4f} late={m(late,"actor_loss"):.4f}')

# Profile distribution
print('\n=== Congestion profile distribution ===')
prof_counts = defaultdict(int)
for r in rows:
    prof_counts[r['congestion_profile']] += 1
total = sum(prof_counts.values())
for p,c in sorted(prof_counts.items()):
    print(f'  {p or "(empty)":<15} {c:4d}  {c/total*100:.1f}%')

# Selectivity sanity: imp_accepted vs imp_refused
print('\n=== Selectivity (mean imp acc vs ref) per mode ===')
for mode in sorted(set(r['policy_mode'] for r in rows)):
    mode_rows = [r for r in rows if r['policy_mode']==mode]
    accs = [num(r['mean_imp_accepted']) for r in mode_rows if num(r['mean_imp_accepted'])]
    refs = [num(r['mean_imp_refused']) for r in mode_rows if num(r['mean_imp_refused'])]
    high = [num(r['accept_rate_high_cong']) for r in mode_rows if num(r['accept_rate_high_cong'])]
    low  = [num(r['accept_rate_low_cong'])  for r in mode_rows if num(r['accept_rate_low_cong'])]
    print(f'  {mode:<16}'
          f' imp_acc={st.mean(accs):.3f}'
          f' imp_ref={st.mean(refs) if refs else 0:.3f}'
          f'  delta={st.mean(accs)-(st.mean(refs) if refs else 0):+.3f}'
          f'  acc_high_cong={st.mean(high) if high else 0:.3f}'
          f'  acc_low_cong={st.mean(low) if low else 0:.3f}')
