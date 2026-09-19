#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2026 Modelon AB
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
Consistency checks for Model Exchange FMUs, usable as a library or from the
command line::

    python -m pyfmi.checks jacobian model.fmu [--time T ...] [--threshold 1e-2]

check_jacobian(): compares the FMU's directional derivatives against coloured
finite differences (forward and central) at one or more points on a trajectory
and reports entries that the directional derivatives get wrong or leave out.
"""

import argparse
import re
import sys

import numpy as np


def _dense(A):
    return A.toarray() if hasattr(A, "toarray") else np.asarray(A, dtype=float)


def _jacobian(model, mode):
    """PyFMI's coloured Jacobian in one mode: 0 = directional derivatives,
    True = forward differences, 2 = central differences."""
    prev = model.force_finite_differences
    model.force_finite_differences = mode
    try:
        return _dense(model._get_A(use_structure_info=True, add_diag=True))
    finally:
        model.force_finite_differences = prev


def _state_kind(name):
    """'a.b.c[3]' -> 'c' (the last name component without indices), used to group rows."""
    return re.sub(r"\[[^\]]*\]", "", name).rsplit(".", 1)[-1]


def check_jacobian(model, times=(), threshold=1e-2, rtol=1e-6, solver="CVode", n_worst=10):
    """
    Compares the FMU's directional derivatives (DD) with coloured finite
    differences at the current state and, optionally, at further points along a
    simulated trajectory.

    At each point three Jacobians of the continuous states are evaluated with
    PyFMI's colouring: DD, forward differences (FD) and central differences (CD).
    Entries where FD and CD disagree by more than 10 % are treated as unreliable
    for finite differences (a kink or a discontinuity next to the point) and are
    left out of the verdict. Of the remaining entries the report lists those that
    the FMU's own dependency structure declares but the DD leaves at zero while
    CD is clearly nonzero ("missing"), those that differ by more than 10 %
    ("wrong"), and the relative Frobenius norm ||DD - CD|| / ||CD||. The check
    fails when that norm exceeds 'threshold' at any point.

    Parameters::

        model --
            A loaded FMUModelME2 or FMUModelME3. If it has not been initialized
            the check initializes it (setup_experiment / initialize / event
            iteration / continuous-time mode).

        times --
            Additional points in time at which to compare; the model is
            simulated to each of them with 'solver' at 'rtol' (default: only
            the current state, i.e. t = start time after initialization).

        threshold --
            Relative norm above which a point is reported as failed.
            Default: 1e-2

        n_worst --
            Number of worst entries listed per point.

    Returns::

        A dict with 'ok' (bool), 'provides_dd', 'nx' and 'points', a list of
        one dict per point with the numbers above and the worst entries as
        (row name, column name, dd, fd, cd, declared).
    """
    if not model.get_capability_flags().get("providesDirectionalDerivatives", False):
        return {"ok": True, "provides_dd": False, "nx": model.get_ode_sizes()[0], "points": [],
                "message": "The FMU does not provide directional derivatives; nothing to compare."}

    if model.time is None:
        if hasattr(model, "setup_experiment"):      # FMI 2
            model.setup_experiment(tolerance=rtol)
            model.initialize()
        else:                                        # FMI 3
            model.initialize(tolerance=rtol)
        model.event_update()
        model.enter_continuous_time_mode()

    states = list(model.get_states_list().keys())
    nx = len(states)
    index = {s: i for i, s in enumerate(states)}
    declared = np.zeros((nx, nx), dtype=bool)
    dep, _ = model.get_derivatives_dependencies()
    for i, (_, cols) in enumerate(dep.items()):
        for s in cols:
            if s in index:
                declared[i, index[s]] = True
    np.fill_diagonal(declared, True)   # PyFMI adds the diagonal to the pattern

    result = {"ok": True, "provides_dd": True, "nx": nx, "points": []}
    first = True
    for t_target in [None] + sorted(t for t in times if t is not None):
        if t_target is not None and t_target > model.time:
            opts = model.simulate_options()
            opts["solver"] = solver
            opts["result_handling"] = None
            opts["ncp"] = 0
            opts["initialize"] = False
            opts[solver + "_options"]["rtol"] = rtol
            opts[solver + "_options"]["verbosity"] = 50   # QUIET: no run statistics on stdout
            model.simulate(start_time=model.time, final_time=t_target, options=opts)
        elif t_target is not None and not first:
            continue
        first = False

        t = model.time
        x = model.continuous_states.copy()
        jac = {}
        for key, mode in (("dd", 0), ("fd", True), ("cd", 2)):
            model.time = t
            model.continuous_states = x
            model.get_derivatives()
            jac[key] = _jacobian(model, mode)
        model.time = t
        model.continuous_states = x
        model.get_derivatives()
        A_dd, A_fd, A_cd = jac["dd"], jac["fd"], jac["cd"]

        scale = np.abs(A_cd).max() if A_cd.size else 1.0
        tiny = 1e-8 * max(scale, 1e-300)
        nonzero = (np.abs(A_cd) > tiny) | (np.abs(A_dd) > tiny)
        rel_fd_cd = np.abs(A_fd - A_cd) / (np.abs(A_cd) + tiny)
        unreliable = nonzero & (rel_fd_cd > 0.1)
        usable = nonzero & ~unreliable
        rel_dd_cd = np.abs(A_dd - A_cd) / (np.abs(A_cd) + tiny)
        missing = usable & declared & (np.abs(A_dd) <= tiny) & (np.abs(A_cd) > 1e3 * tiny)
        wrong = usable & (np.abs(A_dd) > tiny) & (rel_dd_cd > 0.1)
        undeclared = (np.abs(A_dd) > 1e3 * tiny) & ~declared
        norm_cd = np.linalg.norm(A_cd[usable]) if usable.any() else 0.0
        rel_norm = (np.linalg.norm((A_dd - A_cd)[usable]) / norm_cd) if norm_cd > 0 else 0.0

        score = np.where(usable, np.abs(A_dd - A_cd) / (np.abs(A_cd) + 1e-3 * scale), 0.0)
        worst = []
        for flat in np.argsort(-score, axis=None)[:n_worst]:
            i, j = np.unravel_index(flat, score.shape)
            if score[i, j] <= 0.0:
                break
            worst.append((states[i], states[j], float(A_dd[i, j]), float(A_fd[i, j]), float(A_cd[i, j]), bool(declared[i, j])))

        rows_missing = {}
        for i in np.where(missing.any(axis=1))[0]:
            k = _state_kind(states[i])
            rows_missing[k] = rows_missing.get(k, 0) + 1

        point = {"time": float(t), "nonzero": int(nonzero.sum()), "dd_nonzero": int((np.abs(A_dd) > tiny).sum()),
                 "cd_nonzero": int((np.abs(A_cd) > tiny).sum()), "declared": int(declared.sum()),
                 "unreliable_fd": int(unreliable.sum()), "missing": int(missing.sum()), "wrong": int(wrong.sum()),
                 "undeclared": int(undeclared.sum()), "rel_norm": float(rel_norm), "ok": bool(rel_norm <= threshold),
                 "missing_rows_by_kind": rows_missing, "worst": worst}
        result["points"].append(point)
        result["ok"] = result["ok"] and point["ok"]
    return result


def format_jacobian_report(res, name=""):
    lines = ["Jacobian check%s: %s" % ((" of " + name) if name else "", "OK" if res["ok"] else "FAILED")]
    if not res.get("provides_dd", True):
        lines.append("  " + res["message"])
        return "\n".join(lines)
    lines.append("  nx = %d, declared dependencies (incl. diagonal) = %d" % (res["nx"], res["points"][0]["declared"] if res["points"] else 0))
    for p in res["points"]:
        lines.append("  t = %-10g %s  ||DD-CD||/||CD|| = %.2e   nonzero: DD %d, CD %d;  missing in DD %d, wrong %d, "
                     "undeclared in DD %d, FD-unreliable (kinks) %d" % (
                         p["time"], "ok    " if p["ok"] else "FAILED", p["rel_norm"], p["dd_nonzero"], p["cd_nonzero"],
                         p["missing"], p["wrong"], p["undeclared"], p["unreliable_fd"]))
        if p["missing_rows_by_kind"]:
            lines.append("      rows with missing DD entries, by state kind: " +
                         ", ".join("%s: %d" % kv for kv in sorted(p["missing_rows_by_kind"].items(), key=lambda kv: -kv[1])))
        for (r, c, dd, fd, cd, decl) in p["worst"]:
            lines.append("      d(%s)/d(%s): DD %.6g  FD %.6g  CD %.6g%s" % (r, c, dd, fd, cd, "" if decl else "  (not declared)"))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pyfmi.checks", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jacobian", help="directional derivatives vs finite differences")
    j.add_argument("fmu")
    j.add_argument("--time", type=float, nargs="*", default=[], help="additional points in time to compare at (simulated to with CVode)")
    j.add_argument("--threshold", type=float, default=1e-2, help="relative norm above which the check fails (default 1e-2)")
    j.add_argument("--rtol", type=float, default=1e-6)
    j.add_argument("--worst", type=int, default=10, help="number of worst entries to list per point")
    j.add_argument("--log-level", type=int, default=2)
    args = ap.parse_args(argv)

    from pyfmi import load_fmu
    model = load_fmu(args.fmu, log_level=args.log_level)
    res = check_jacobian(model, times=args.time, threshold=args.threshold, rtol=args.rtol, n_worst=args.worst)
    print(format_jacobian_report(res, args.fmu))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
