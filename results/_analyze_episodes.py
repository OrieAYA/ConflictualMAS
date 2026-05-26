import csv, statistics, collections

def analyze(path, label):
    print(f"\n========== {label} ==========")
    print(f"file: {path}")
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    print(f"total episode rows: {len(rows)}")

    groups = collections.defaultdict(list)
    for r in rows:
        key = (r['city'], r['phase'], r['policy_mode'])
        groups[key].append(r)

    out = []
    for (city, phase, mode), grp in groups.items():
        n = len(grp)
        def col(name):
            return [float(r[name]) for r in grp]
        thr = statistics.mean(col('throughput_rate'))
        acc = statistics.mean(col('accept_rate'))
        lat = statistics.mean(col('latency_mean'))
        out.append((phase, mode, city, n, thr, acc, lat))

    phase_order = {'train':0, 'eval':1, 'stress':2, 'generalize':3}
    out.sort(key=lambda x: (phase_order.get(x[0], 99), x[1], x[2]))

    current_phase = None
    for phase, mode, city, n, thr, acc, lat in out:
        if phase != current_phase:
            print(f"\n--- phase={phase} ---")
            print(f"  {'mode':<18} {'city':<22} {'n':>3}  {'thr':>7}  {'acc':>7}  {'lat':>9}")
            current_phase = phase
        print(f"  {mode:<18} {city:<22} {n:>3}  {thr:>7.4f}  {acc:>7.4f}  {lat:>9.2f}")

    print(f"\n--- {label} : cross-city mean per (phase, mode) ---")
    crossm = collections.defaultdict(list)
    for phase, mode, city, n, thr, acc, lat in out:
        crossm[(phase, mode)].append((thr, acc, lat))
    print(f"  {'phase':<12} {'mode':<18} {'thr':>7}  {'acc':>7}  {'lat':>9}")
    rows_x = []
    for (phase, mode), vals in crossm.items():
        thr_m = statistics.mean(v[0] for v in vals)
        acc_m = statistics.mean(v[1] for v in vals)
        lat_m = statistics.mean(v[2] for v in vals)
        rows_x.append((phase, mode, thr_m, acc_m, lat_m))
    rows_x.sort(key=lambda x: (phase_order.get(x[0],99), -x[2]))
    for phase, mode, thr_m, acc_m, lat_m in rows_x:
        print(f"  {phase:<12} {mode:<18} {thr_m:>7.4f}  {acc_m:>7.4f}  {lat_m:>9.2f}")

analyze("results/Global Analysis preceding architecture/mc_format_A_052510h/episodes_seed42.csv", "FORMAT A (force_assign)")
analyze("results/Global Analysis preceding architecture/mc_format_B_052510h/episodes_seed42.csv", "FORMAT B (no force_assign)")