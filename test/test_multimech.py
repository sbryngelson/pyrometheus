"""Multi-mechanism Fortran codegen: N=1 identity and N>1 selection."""
import os
import subprocess
import tempfile

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
    # specialized form: no runtime selection machinery
    assert "mech_id" not in actual
    assert "select case (mech_id)" not in actual
    nsp = _sol(mechname).n_species
    assert f"integer, parameter :: num_species = {nsp}" in actual
    # ...but num_species_max is exported here too, so MFC sizes arrays with one
    # spelling in both forms.
    assert f"integer, parameter :: num_species_max = {nsp}" in actual


def test_multi_mechanism_preamble():
    """N>1 emits the selection scaffolding; N=1 must not."""
    src = pyro.FortranCodeGenerator.generate(
        "m_thermochem", [_sol("uiuc"), _sol("sandiego")],
        pyro.CodeGenerationOptions())

    # uiuc has 7 species, sandiego 9 -> the bound is the max
    assert "integer, parameter :: num_species_max = 9" in src
    assert "integer :: num_species" in src
    assert "integer :: mech_id" in src
    assert "subroutine set_mechanism" in src

    single = reference_source("uiuc")
    assert "mech_id" not in single
    assert "select case (mech_id)" not in single
    assert "integer, parameter :: num_species = 7" in single
    assert "integer, parameter :: num_species_max = 7" in single


def test_routine_body_branches_on_mech_id():
    """Each arm inlines expressions; no arm may call another device routine."""
    src = pyro.FortranCodeGenerator.generate(
        "m_thermochem", [_sol("uiuc"), _sol("sandiego")],
        pyro.CodeGenerationOptions())

    start = src.index("subroutine get_species_enthalpies_rt")
    body = src[start:src.index("end subroutine get_species_enthalpies_rt")]

    assert "select case (mech_id)" in body
    assert body.count("case (") >= 2
    assert "dimension(num_species_max)" in body
    # Arms must inline expressions, never call out -- a device routine calling
    # another device routine faults at runtime under CCE OpenMP.
    assert "call " not in body


def _build_and_run(source, driver, args, tmp):
    """Compile the generated module against a driver and return its numbers."""
    open(os.path.join(tmp, "m_thermochem.f90"), "w").write(source)
    open(os.path.join(tmp, "drv.f90"), "w").write(open(driver).read())
    subprocess.run(["gfortran", "-cpp", "-ffree-line-length-none", "-O1",
                    "m_thermochem.f90", "drv.f90", "-o", "drv"],
                   cwd=tmp, check=True)
    out = subprocess.run(["./drv"] + args, cwd=tmp, capture_output=True,
                         text=True, check=True)
    return [float(x) for x in out.stdout.split()]


BUNDLE = ["uiuc", "sandiego", "hong"]


@pytest.mark.parametrize("mechname", BUNDLE)
def test_arm_matches_single_mechanism(mechname):
    """Each arm must reproduce the dedicated single-mechanism generator exactly."""
    here = os.path.dirname(__file__)
    multi = pyro.FortranCodeGenerator.generate(
        "m_thermochem", [_sol(m) for m in BUNDLE],
        pyro.CodeGenerationOptions())

    with tempfile.TemporaryDirectory() as w:
        got = _build_and_run(multi, os.path.join(here, "multimech_driver.f90"),
                             [mechname], w)
    with tempfile.TemporaryDirectory() as w:
        expected = _build_and_run(reference_source(mechname),
                                  os.path.join(here, "singlemech_driver.f90"),
                                  [], w)

    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        assert abs(a - b) <= 1e-13 * max(1.0, abs(b))


def test_runtime_mechanism_form_for_single_mechanism():
    """A single mechanism can be emitted in the runtime-selectable form, so MFC
    can adopt runtime num_species before any bundling happens."""
    src = pyro.FortranCodeGenerator.generate(
        "m_thermochem", _sol("sandiego"),
        pyro.CodeGenerationOptions(runtime_mechanism=True))
    assert "integer, parameter :: num_species_max = 9" in src
    assert "integer :: num_species = 9" in src
    assert "select case (mech_id)" in src
    assert "subroutine set_mechanism" in src
    # and the default is still the specialized form
    assert "mech_id" not in reference_source("sandiego")
