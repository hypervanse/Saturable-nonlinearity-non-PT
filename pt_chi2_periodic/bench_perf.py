"""
bench_perf.py -- measurements behind the performance paper (manuscript_perf/).  Run it on an IDLE
machine: every number is wall-clock time.  Output: results/perf.json

  A  dense LAPACK kernels vs matrix size n and BLAS thread count:
       eigvals (zgeev: Hessenberg reduction + shifted QR), inv (zgetrf + zgetri), solve (zgesv, 1 rhs)
  B  throughput of 4 independent eigensolves (n = 2000):
       1 process x 4 threads,  2 processes x 2 threads,  4 processes x 1 thread
  C  anatomy of one convergence-study row (L = 40 pi, h = 0.25, 3 threads as in production):
       wall time of each stage, from the stored edge shot to the two spectra
  D  interpreter-bound kernels (1 thread):
       RKF45 edge shot (time per accepted step), scalar vs vectorized right-hand side,
       split-step time per step vs grid size M, and batching K independent runs into one array

The number of BLAS threads is fixed when numpy is imported, so the driver re-launches this
script as a child process per setting:  python3 bench_perf.py <part> <threads>
"""
import json
import os
import subprocess
import sys
import time

PART = sys.argv[1] if len(sys.argv) > 1 else 'driver'
THREADS = int(sys.argv[2]) if len(sys.argv) > 2 else 1


def child(part, threads, extra=()):
    """run `python3 bench_perf.py part threads` with the BLAS thread count fixed; return its JSON"""
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads))
    out = subprocess.run([sys.executable, __file__, part, str(threads), *extra], env=env,
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().split('\n')[-1])


if PART == 'driver':
    t0 = time.time()
    res = dict(machine=dict(cpus=os.cpu_count(), cpu=open('/proc/cpuinfo').read().split('model name')[1]
                            .split('\n')[0].strip(': \t')))
    import numpy as np
    res['machine']['numpy'] = np.__version__
    print('A: dense kernels vs n', flush=True)
    res['A'] = {t: child('A', t) for t in (1, 2, 3, 4)}
    print('B: throughput', flush=True)
    res['B'] = {}
    res['B']['1x4'] = child('B1', 4)                       # one process, 4 eigensolves in series, 4 threads
    for nproc, thr in ((2, 2), (4, 1)):                     # several processes at once
        env = dict(os.environ, OMP_NUM_THREADS=str(thr), OPENBLAS_NUM_THREADS=str(thr))
        ts = time.time()
        ps = [subprocess.Popen([sys.executable, __file__, 'B1', str(thr), str(4 // nproc)], env=env,
                               stdout=subprocess.PIPE, text=True) for _ in range(nproc)]
        outs = [json.loads(p.communicate()[0].strip().split('\n')[-1]) for p in ps]
        res['B'][f'{nproc}x{thr}'] = dict(wall=time.time() - ts, per_proc=outs)
    print('C: convergence-row anatomy', flush=True)
    res['C'] = child('C', 3)
    print('D: interpreter-bound kernels', flush=True)
    res['D'] = child('D', 1)
    # per-row wall times of the production convergence study (cumulative [..s] stamps in its log)
    import re
    if os.path.exists('logs/convergence.log'):
        rows = []
        for line in open('logs/convergence.log'):
            m = re.match(r'L=(\d+)pi h=([\d.]+) al=([\d.]+)\s+b=([-\d.]+).*\[(\d+)s\]', line)
            if m:
                rows.append(dict(L_pi=int(m[1]), h=float(m[2]), al=float(m[3]), b=float(m[4]), t_cum=int(m[5])))
        res['conv_rows'] = rows
    res['total_s'] = time.time() - t0
    from io_utils import dump
    dump('results/perf.json', res)
    print(f"all done [{res['total_s']:.0f}s]")
    sys.exit(0)

# ------------------------------------------------------------------------------------ children
import numpy as np  # noqa: E402  (after the thread count is in the environment)

rng = np.random.default_rng(0)
out = {}


def best(f, reps):
    """minimum wall time of `reps` calls (the minimum is the least disturbed by other activity)"""
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return min(ts)


if PART == 'A':
    for n in (250, 500, 1000, 1500, 2000, 2800, 4024):
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        b = rng.standard_normal(n) + 0j
        reps = 3 if n <= 1000 else 1
        out[n] = dict(eig=best(lambda: np.linalg.eigvals(A), reps),
                      inv=best(lambda: np.linalg.inv(A), reps),
                      solve=best(lambda: np.linalg.solve(A, b), reps))
        print(n, out[n], file=sys.stderr, flush=True)

elif PART == 'B1':
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 4         # eigensolves done by this process
    n = 2000
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    t = time.perf_counter()
    for _ in range(k):
        np.linalg.eigvals(A)
    out = dict(k=k, n=n, t=time.perf_counter() - t)

elif PART == 'C':
    from shooting import tails, newton_polish, profile_from_shot
    from newton_fd import Grid, solve, continuation, residual_jacobian, pack
    from pipeline import to_grid, solution_at, stability_of
    T = {}
    par0 = dict(V1=1.0, V2=1.0, al=0.0, q=0.0, b=-1.207, x0=np.pi / 2)
    sh0 = json.load(open('results/thirdgap.json'))['shoot']
    t = time.perf_counter(); Tl = tails(par0, 8); T['floquet_tails'] = time.perf_counter() - t
    t = time.perf_counter(); c1, c2 = newton_polish(par0, Tl, sh0['c1'], sh0['c2'], verbose=False)
    T['shoot_polish'] = time.perf_counter() - t
    t = time.perf_counter(); xs, w1, p1, w2, p2, info = profile_from_shot(par0, Tl, c1, c2)
    T['shoot_profile'] = time.perf_counter() - t
    t = time.perf_counter(); g = Grid(L=40 * np.pi, h=0.25, deriv='fourier'); T['grid_D2'] = time.perf_counter() - t
    t = time.perf_counter(); s1, s2 = to_grid(g, dict(xs=xs, w1=w1, p1=p1, w2=w2, p2=p2)); T['spline'] = time.perf_counter() - t
    t = time.perf_counter(); v1, v2, it, res = solve(g, s1, s2, par0); T['newton_al0'] = time.perf_counter() - t
    # one Newton iteration: Jacobian assembly and the dense solve, separately
    t = time.perf_counter(); F, J = residual_jacobian(pack(v1, v2), g, par0); T['jacobian_assembly'] = time.perf_counter() - t
    t = time.perf_counter(); np.linalg.solve(J, F); T['jacobian_solve'] = time.perf_counter() - t
    als = list(np.round(np.arange(0.05, 0.7501, 0.05), 3)) + [0.7919]
    t = time.perf_counter(); br = continuation(g, v1, v2, par0, 'al', als, verbose=False)
    T['alpha_continuation'] = time.perf_counter() - t
    e1, e2 = br[-1]['w1'], br[-1]['w2']
    par = dict(par0, al=0.7919)
    t = time.perf_counter(); e1, e2 = solution_at(g, e1, e2, par, -1.262, db=0.005); T['b_continuation_11'] = time.perf_counter() - t
    pb = dict(par, b=-1.262)
    t = time.perf_counter(); A1 = stability_of(g, dict(w1=e1, w2=e2), pb, g0=1.0); T['eig_g0_1'] = time.perf_counter() - t
    t = time.perf_counter(); stability_of(g, dict(w1=e1, w2=e2), pb, g0=0.25); T['eig_g0_025'] = time.perf_counter() - t
    lam = A1['lam_unst'] if A1['lam_unst'] is not None else 0.05j
    t = time.perf_counter(); stability_of(g, dict(w1=e1, w2=e2), pb, g0=1.0, method='arnoldi',
                                          shifts=[complex(abs(lam.real), lam.imag) + 0.01j])
    T['arnoldi_target'] = time.perf_counter() - t
    out = dict(stages=T, M=g.M, n_stab=4 * g.M, n_newton=J.shape[0], newton_its=it, n_alpha=len(als),
               rk_steps=info['nacc'], rk_rejected=info['nrej'])

elif PART == 'D':
    import math
    from model import make_rhs
    from shooting import tails, edge_shot
    import ssf
    S = json.load(open('results/semiinf.json'))['cases']['case1']
    par = dict(V1=1.0, V2=1.0, al=0.0, q=S['q'], b=S['b_shoot'], x0=0.0)
    Tl = tails(par, S['n_periods'])
    t = time.perf_counter(); D1, D2, xs, ys, info = edge_shot(S['shoot']['c1'], S['shoot']['c2'], par, Tl, 1e-10, 1e-12, 1e8, False)
    tshot = time.perf_counter() - t
    out['rk_shot'] = dict(t=tshot, nacc=info['nacc'], nrej=info['nrej'],
                          us_per_step=1e6 * tshot / (info['nacc'] + info['nrej']))
    # right-hand side: one scalar call (complex Python numbers) vs vectorized over K states
    f = make_rhs(1.0, 1.0, 0.0, S['q'], S['b_shoot'])
    y = (0.1 + 0j, 0.01 + 0j, 0.05 + 0j, 0.0 + 0j)
    nrep = 20000
    t = time.perf_counter()
    for i in range(nrep):
        f(0.3, y)
    out['rhs_scalar_us'] = 1e6 * (time.perf_counter() - t) / nrep

    def rhs_vec(x, w1, p1, w2, p2, V1=1.0, V2=1.0, al=0.0, q=S['q'], b=S['b_shoot']):
        c, s = np.cos(2 * x), np.sin(2 * x)
        return p1, -(V1 * c - b + 1j * al * s) * w1 - 2 * np.conj(w1) * w2, p2, -4 * (V2 * c + q - b) * w2 - 2 * w1 * w1

    out['rhs_vec_us_per_state'] = {}
    for K in (1, 16, 256, 4096):
        W = [np.full(K, v) for v in y]
        r = max(10, 20000 // K)
        tv = best(lambda: [rhs_vec(0.3, *W) for _ in range(r)], 3) / r
        out['rhs_vec_us_per_state'][K] = 1e6 * tv / K
    # split-step: time per Strang step vs M, and achieved flop rate (FFT ~ 5 M log2 M real flops)
    out['ssf'] = {}
    for M in (256, 1024, 4096, 16384, 65536):
        L = 40 * np.pi
        x, k = ssf.make_box(L, M)
        U1 = np.cos(2 * x) + 0.7j * np.sin(2 * x)
        U2 = 2 * np.cos(2 * x)
        gab = ssf.absorber(x, L, 0.15, 1.0)
        dz = 0.01
        E1h, E2h = np.exp(-1j * k ** 2 * dz / 2), np.exp(-0.5j * k ** 2 * dz / 2)
        u1 = 0.1 * np.exp(-x ** 2) + 0j
        u2 = 0.05 * np.exp(-x ** 2) + 0j
        nst = max(20, 200000 // M)

        def run():
            a, b2 = u1, u2
            for _ in range(nst):
                a, b2 = ssf._D_step(a, b2, E1h, E2h)
                a, b2 = ssf._N_step(a, b2, dz, U1, U2, gab)
                a, b2 = ssf._D_step(a, b2, E1h, E2h)
        tstep = best(run, 3) / nst
        flops = 2 * 4 * 5 * M * math.log2(M) + 4 * 40 * M      # 8 complex FFTs + 4 RHS evaluations (~40 flops/point)
        out['ssf'][M] = dict(us_per_step=1e6 * tstep, gflops=flops / tstep / 1e9)
    # batching K independent runs (e.g. noise realizations) into one (K, M) array, M = 1024
    out['ssf_batch'] = {}
    M = 1024
    x, k = ssf.make_box(40 * np.pi, M)
    U1 = np.cos(2 * x) + 0.7j * np.sin(2 * x)
    U2 = 2 * np.cos(2 * x)
    gab = ssf.absorber(x, 40 * np.pi, 0.15, 1.0)
    E1h, E2h = np.exp(-1j * k ** 2 * 0.005), np.exp(-0.5j * k ** 2 * 0.005)
    for K in (1, 4, 16, 64):
        a0 = np.tile(0.1 * np.exp(-x ** 2) + 0j, (K, 1))
        b0 = np.tile(0.05 * np.exp(-x ** 2) + 0j, (K, 1))
        nst = 100

        def runb():
            a, b2 = a0, b0
            for _ in range(nst):
                a, b2 = ssf._D_step(a, b2, E1h, E2h)
                a, b2 = ssf._N_step(a, b2, 0.01, U1, U2, gab)
                a, b2 = ssf._D_step(a, b2, E1h, E2h)
        out['ssf_batch'][K] = 1e6 * best(runb, 3) / nst / K           # us per step per run

print(json.dumps(out, default=lambda o: [o.real, o.imag] if isinstance(o, complex) else float(o)))
