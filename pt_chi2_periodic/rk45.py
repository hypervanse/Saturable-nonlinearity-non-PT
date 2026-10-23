"""
rk45.py -- Runge-Kutta-Fehlberg 4(5) with adaptive step ("interstep refinement").

Thesis, ch. 3: "Runge-Kutta integrations of order 4 and 5. The error in the step
is approximated as the maximal value of the difference between the two
integrations."  That is exactly what is coded below:

  1. one step of size h produces y4 (4th order) and y5 (5th order) from the
     SAME six stage evaluations (Fehlberg's embedded pair);
  2. err = max_i |y5_i - y4_i| / (atol + rtol*|y_i|)   (max-norm, scaled);
  3. err > 1  -> step REJECTED, h is refined and the step is redone from the
                 same point (this is the interstep refinement);
     err <= 1 -> step accepted, solution advanced with y5 (local extrapolation);
  4. next h = h * 0.9 * err^(-1/5), clipped to [0.2, 5] x h.
     (exponent 1/5 because the local error of the 4th-order formula ~ h^5)

State is a tuple of python complex numbers (4 entries for the shooting problem):
for such tiny systems pure-python scalar arithmetic beats numpy array overhead.
Works in either direction (x1 < x0 integrates backwards, as edge shooting does).
"""
import math

# Fehlberg tableau ----------------------------------------------------------
_C2, _C3, _C4, _C5, _C6 = 1 / 4, 3 / 8, 12 / 13, 1.0, 1 / 2
_A21 = 1 / 4
_A31, _A32 = 3 / 32, 9 / 32
_A41, _A42, _A43 = 1932 / 2197, -7200 / 2197, 7296 / 2197
_A51, _A52, _A53, _A54 = 439 / 216, -8.0, 3680 / 513, -845 / 4104
_A61, _A62, _A63, _A64, _A65 = -8 / 27, 2.0, -3544 / 2565, 1859 / 4104, -11 / 40
# 5th-order weights (used to advance)
_B1, _B3, _B4, _B5, _B6 = 16 / 135, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55
# error weights E = b5 - b4  (b4 = 25/216, 0, 1408/2565, 2197/4104, -1/5, 0)
_E1, _E3, _E4, _E5, _E6 = 1 / 360, -128 / 4275, -2197 / 75240, 1 / 50, 2 / 55


def rkf45(f, x0, y0, x1, h0=1e-2, rtol=1e-10, atol=1e-13,
          hmax=0.25, hmin=1e-12, ymax=math.inf, store=True, maxsteps=10**6):
    """
    Integrate y' = f(x, y) from x0 to x1.

    Returns (xs, ys, info):
      xs, ys : lists of accepted mesh points / states (only endpoints if store=False)
      info   : dict(nacc, nrej, blowup) -- blowup=True if max|y| exceeded ymax,
               in which case integration stopped early at xs[-1].
    """
    n = len(y0)
    sgn = 1.0 if x1 >= x0 else -1.0
    h = sgn * min(abs(h0), hmax)
    x = x0
    y = tuple(complex(v) for v in y0)
    xs, ys = [x], [y]
    nacc = nrej = 0
    blowup = False
    R = range(n)

    for _ in range(maxsteps):
        if sgn * (x1 - x) <= 0.0:
            break
        if sgn * (x + h - x1) > 0.0:          # clip last step to land exactly on x1
            h = x1 - x

        # six stages (each stage = state increment built from previous slopes)
        k1 = f(x, y)
        k2 = f(x + _C2 * h, tuple(y[i] + h * _A21 * k1[i] for i in R))
        k3 = f(x + _C3 * h, tuple(y[i] + h * (_A31 * k1[i] + _A32 * k2[i]) for i in R))
        k4 = f(x + _C4 * h, tuple(y[i] + h * (_A41 * k1[i] + _A42 * k2[i] + _A43 * k3[i]) for i in R))
        k5 = f(x + _C5 * h, tuple(y[i] + h * (_A51 * k1[i] + _A52 * k2[i] + _A53 * k3[i]
                                               + _A54 * k4[i]) for i in R))
        k6 = f(x + _C6 * h, tuple(y[i] + h * (_A61 * k1[i] + _A62 * k2[i] + _A63 * k3[i]
                                               + _A64 * k4[i] + _A65 * k5[i]) for i in R))

        # 5th-order solution and embedded error estimate |y5 - y4|
        y5 = tuple(y[i] + h * (_B1 * k1[i] + _B3 * k3[i] + _B4 * k4[i] + _B5 * k5[i]
                               + _B6 * k6[i]) for i in R)
        err = 0.0
        for i in R:
            e = abs(h * (_E1 * k1[i] + _E3 * k3[i] + _E4 * k4[i] + _E5 * k5[i] + _E6 * k6[i]))
            sc = atol + rtol * max(abs(y[i]), abs(y5[i]))
            err = max(err, e / sc)

        if err <= 1.0:                         # ---- accept
            x += h
            y = y5
            nacc += 1
            if store:
                xs.append(x)
                ys.append(y)
            if max(abs(v) for v in y) > ymax:  # solution ran away: stop, flag it
                blowup = True
                break
        else:                                  # ---- reject: refine and redo
            nrej += 1

        # step-size controller (same formula after accept or reject)
        fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * err ** -0.2))
        h = sgn * min(hmax, abs(h) * fac)
        if abs(h) < hmin:
            raise RuntimeError(f"rkf45: step underflow at x={x}")

    if not store:
        xs, ys = [x], [y]
    return xs, ys, dict(nacc=nacc, nrej=nrej, blowup=blowup)
