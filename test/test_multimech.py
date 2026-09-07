"""Multi-mechanism Fortran codegen: N=1 identity and N>1 selection."""
import os

import cantera as ct
import pytest

import pyrometheus as pyro

MECH_DIR = os.path.join(os.path.dirname(__file__), "mechs")


def _sol(mechname):
    return ct.Solution(os.path.join(MECH_DIR, f"{mechname}.yaml"), "gas")


def reference_source(mechname, **opt_kwargs):
    """Single-mechanism Fortran source, via the long-standing single-sol API."""
    return pyro.FortranCodeGenerator.generate(
        "m_thermochem", _sol(mechname),
        pyro.CodeGenerationOptions(**opt_kwargs))


@pytest.mark.parametrize("mechname", ["uiuc", "sandiego", "hong"])
def test_single_mechanism_list_form_is_identical(mechname):
    """Passing a one-element mechanism list must emit exactly what passing a bare
    Solution emits. This is the invariant that keeps MFC's case-optimized path and
    every existing consumer untouched."""
    expected = reference_source(mechname)
    actual = pyro.FortranCodeGenerator.generate(
        "m_thermochem", [_sol(mechname)], pyro.CodeGenerationOptions())
    assert actual == expected
