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

"""Tests for pyfmi.checks (FMU consistency checks)."""

import numpy as np
import pytest

from pyfmi.checks import check_jacobian, format_jacobian_report, main


@pytest.fixture
def vanderpol():
    from tests.utils import get_fmi3_reference_fmu
    return get_fmi3_reference_fmu("VanDerPol")


@pytest.fixture
def dahlquist():
    from tests.utils import get_fmi3_reference_fmu
    return get_fmi3_reference_fmu("Dahlquist")


class TestCheckJacobian:
    def test_consistent(self, vanderpol):
        res = check_jacobian(vanderpol, times=[1.0, 5.0])
        assert res["ok"] and res["provides_dd"]
        assert [p["time"] for p in res["points"]] == [0.0, 1.0, 5.0]
        for p in res["points"]:
            assert p["rel_norm"] < 1e-6
            assert p["missing"] == 0 and p["wrong"] == 0 and p["undeclared"] == 0
            assert p["dd_nonzero"] == 3      # d(x0)/d(x1), d(x1)/d(x0), d(x1)/d(x1); d(x0)/d(x0) = 0
        assert "OK" in format_jacobian_report(res, "VanDerPol")

    def test_missing_entries_detected(self):
        """Directional derivatives that leave declared entries at zero are reported."""
        from tests.utils import get_fmi3_reference_fmu
        from pyfmi.fmi3 import FMUModelME3

        class ZeroDD(FMUModelME3):
            def get_directional_derivative(self, var_ref, func_ref, v):
                return np.zeros(len(func_ref))

        vanderpol = get_fmi3_reference_fmu("VanDerPol", model_class=ZeroDD)
        res = check_jacobian(vanderpol, times=[1.0])
        assert not res["ok"]
        for p in res["points"]:
            assert p["rel_norm"] > 0.5
            assert p["dd_nonzero"] == 0
            assert p["missing"] == 3
            assert p["missing_rows_by_kind"] == {"x0": 1, "x1": 1}
        assert any(w[2] == 0.0 and abs(w[4]) > 0.1 for w in res["points"][0]["worst"])
        assert "FAILED" in format_jacobian_report(res)

    def test_no_directional_derivatives(self, dahlquist):
        res = check_jacobian(dahlquist)
        assert res["ok"] and not res["provides_dd"] and res["points"] == []
        assert "does not provide" in format_jacobian_report(res)

    def test_state_restored(self, vanderpol):
        vanderpol.initialize(); vanderpol.event_update(); vanderpol.enter_continuous_time_mode()
        x = vanderpol.continuous_states.copy()
        check_jacobian(vanderpol)
        assert vanderpol.time == 0.0
        np.testing.assert_array_equal(vanderpol.continuous_states, x)
        assert vanderpol.force_finite_differences is False

    def test_cli(self, capsys):
        from tests.utils import FMI3_REF_FMU_PATH
        status = main(["jacobian", str(FMI3_REF_FMU_PATH / "VanDerPol.fmu"), "--time", "1.0", "--log-level", "0"])
        assert status == 0
        assert "Jacobian check" in capsys.readouterr().out
