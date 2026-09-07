"""
Fortran code generation
-----------------------

.. autoclass:: FortranCodeGenerator
"""

import os
import shlex
from functools import partial
from numbers import Number

import numpy as np  # noqa: F401
import cantera as ct
import pymbolic.primitives as p
from mako.template import Template
import pyrometheus.chem_expr
from pymbolic.mapper.stringifier import (
        StringifyMapper, PREC_NONE, PREC_CALL, PREC_PRODUCT)

from . import CodeGenerator, CodeGenerationOptions


file_extension = "f90"


# {{{ code generation helpers

def pad_fortran(line, width):
    line += " " * (width - 1 - len(line))
    line += "&"
    return line


def wrap_line_base(line, level=0, width=80, indentation="    ",
                   pad_func=lambda string, amount: string,
                   lex_func=None):
    """
    The input is a line of code at the given indentation level. Return the list
    of lines that results from wrapping the line to the given width. Lines
    subsequent to the first line in the returned list are padded with extra
    indentation. The initial indentation level is not included in the input or
    output lines.

    The `pad_func` argument is a function that adds line continuations. The
    `lex_func` argument returns the list of tokens in the line.
    """
    if lex_func is None:
        lex_func = partial(shlex.split, posix=False)

    tokens = lex_func(line)
    resulting_lines = []
    at_line_start = True
    indentation_len = len(level * indentation)
    current_line = ""
    padding_width = width - indentation_len
    for index, word in enumerate(tokens):
        has_next_word = index < len(tokens) - 1
        word_len = len(word)
        if not at_line_start:
            next_len = indentation_len + len(current_line) + 1 + word_len
            if next_len < width or (not has_next_word and next_len == width):
                # The word goes on the same line.
                current_line += " " + word
            else:
                # The word goes on the next line.
                resulting_lines.append(pad_func(current_line, padding_width))
                at_line_start = True
                current_line = indentation
        if at_line_start:
            current_line += word
            at_line_start = False
    resulting_lines.append(current_line)
    return resulting_lines


def count_leading_spaces(s):
    n = 0
    while n < len(s) and s[n] == " ":
        n += 1
    return n


def wrap_code(s, indent=4):
    lines = s.split("\n")
    result_lines = []
    for ln in lines:
        nspaces = count_leading_spaces(ln)
        level, remainder = divmod(nspaces, indent)

        if remainder != 0:
            raise ValueError(f"indentation of '{ln}' is not a multiple of "
                    f"{indent}")

        result_lines.extend(
                (level * indent) * " " + subln
                for subln in
                wrap_line_base(ln, level=level, indentation=" "*indent,
                    pad_func=pad_fortran))

    return "\n".join(result_lines)


def float_to_fortran(num):
    result = f"{num}".replace("e", "d")
    if "d" not in result:
        result = result+"d0"
    if num < 0:
        result = "(%s)" % result
    return result


def str_np_inner(ary):
    if isinstance(ary, Number):
        return float_to_fortran(ary)
    elif ary.shape:
        return "(%s)" % (", ".join(str_np_inner(ary_i) for ary_i in ary))
    raise TypeError("invalid argument to str_np_inner")


def str_np(ary):
    return ", ".join(float_to_fortran(entry) for entry in ary)

# }}}


# {{{ fortran expression generation

class FortranExpressionMapper(StringifyMapper):
    """Converts expressions to Fortran code."""

    def map_constant(self, expr, enclosing_prec):
        if isinstance(expr, bool):
            if expr:
                return ".true."
            else:
                return ".false."
        else:
            return float_to_fortran(expr)

    def map_variable(self, expr, enclosing_prec):
        return expr.name

    def map_lookup(self, expr, enclosing_prec):
        return self.parenthesize_if_needed(
                self.format("%s%%%s",
                    self.rec(expr.aggregate, PREC_CALL),
                    expr.name),
                enclosing_prec, PREC_CALL)

    def map_subscript(self, expr, enclosing_prec):
        def get_base_and_indices(expr):
            if not hasattr(expr, "aggregate") or not hasattr(expr, "index"):
                return expr, []

            # Get current level indices
            if isinstance(expr.index, tuple):
                current_indices = [self.rec(i, PREC_NONE) for i in expr.index]
            else:
                current_indices = [self.rec(expr.index, PREC_NONE)]

            # Only recurse if aggregate is another subscript
            if (hasattr(expr.aggregate, "aggregate")
            and hasattr(expr.aggregate, "index")):
                base, prev_indices = get_base_and_indices(expr.aggregate)
                return base, prev_indices + current_indices
            else:
                return expr.aggregate, current_indices

        # Get base array and all indices
        base_array, all_indices = get_base_and_indices(expr)

        # Convert float indices (ending with 'd0') to integers and add 1
        def convert_index(idx):
            idx_str = str(idx)
            if idx_str.endswith("d0"):
                # Remove 'd0' suffix and convert to int
                num = int(float(idx_str.replace("d0", "")))
                return str(num + 1)
            try:
                # Try to convert to int and add 1
                return str(int(idx_str) + 1)
            except ValueError:
                # If it's not a simple number, wrap in a +1
                return f"({idx_str} + 1)"

        # Format indices, converting floats to integers and adding 1
        index_str = ", ".join(convert_index(idx) for idx in all_indices)

        # Format the final expression
        return self.parenthesize_if_needed(
            self.format("%s(%s)", self.rec(base_array, PREC_CALL), index_str),
            enclosing_prec,
            PREC_CALL
        )

    def map_product(self, expr, enclosing_prec, *args, **kwargs):
        # This differs from the superclass only by adding spaces
        # around the operator, which provide an opportunity for
        # line breaking.
        return self.parenthesize_if_needed(
                self.join_rec(" * ", expr.children, PREC_PRODUCT, *args, **kwargs),
                enclosing_prec, PREC_PRODUCT)

    def map_logical_not(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_UNARY
        return self.parenthesize_if_needed(
                ".not. " + self.rec(expr.child, PREC_UNARY),
                enclosing_prec, PREC_UNARY)

    def map_logical_or(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_LOGICAL_OR
        return self.parenthesize_if_needed(
                self.join_rec(
                    " .or. ", expr.children, PREC_LOGICAL_OR),
                enclosing_prec, PREC_LOGICAL_OR)

    def map_logical_and(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_LOGICAL_AND
        return self.parenthesize_if_needed(
                self.join_rec(
                    " .and. ", expr.children, PREC_LOGICAL_AND),
                enclosing_prec, PREC_LOGICAL_AND)

    def map_if(self, expr, enclosing_prec):
        return self.format(
            "merge(%s)" % self.join_rec(", ",
                                        [expr.then, expr.else_,
                                         expr.condition],
                                        PREC_NONE))
# }}}


# {{{ module template

module_tpl = Template("""
#ifndef PYROMETHEUS_CALLER_INDEXING
#define PYROMETHEUS_CALLER_INDEXING 0
#endif

${gpu_routine}

module ${module_name}

    implicit none

    integer, parameter :: sp = selected_real_kind(6,37)   ! Single precision
    integer, parameter :: dp = selected_real_kind(15,307) ! Double precision

    %if multi:
    integer, parameter :: num_species_max = ${num_species_max}
    integer, parameter :: num_reactions_max = ${num_reactions_max}
    integer, parameter :: num_elements_max = ${num_elements_max}
    integer, parameter :: num_mechanisms = ${len(mechs)}
    integer :: num_species = ${mechs[0]["sol"].n_species}
    integer :: num_reactions = ${mechs[0]["sol"].n_reactions}
    integer :: num_elements = ${mechs[0]["sol"].n_elements}
    integer :: num_falloff = ${len(mechs[0]["falloff"])}
    integer :: mech_id = 1
    ${real_type}, parameter :: one_atm = ${float_to_fortran(ct.one_atm)}
    ${real_type}, parameter :: gas_constant = ${float_to_fortran(ct.gas_constant)}
    ! Filled by set_mechanism, so every reference in the routine bodies below is
    ! spelled exactly as it is in single-mechanism output.
    ${real_type} :: molecular_weights(${num_species_max})
    ${real_type} :: inv_molecular_weights(${num_species_max})
    character(len=12) :: species_names(${num_species_max})
    character(len=4) :: element_names(${num_elements_max})
    %else:
    integer, parameter :: num_elements = ${sol.n_elements}
    integer, parameter :: num_species = ${sol.n_species}
    integer, parameter :: num_reactions = ${sol.n_reactions}
    integer, parameter :: num_falloff = ${len(falloff_reactions)}
    ${real_type}, parameter :: one_atm = ${float_to_fortran(ct.one_atm)}
    ${real_type}, parameter :: gas_constant = ${float_to_fortran(ct.gas_constant)}
    ${real_type}, parameter :: molecular_weights(${sol.n_species}) = &
        (/ ${str_np(sol.molecular_weights)} /)
    ${real_type}, parameter :: inv_molecular_weights(${sol.n_species}) = &
        (/ ${str_np(1/sol.molecular_weights)} /)

    character(len=12), parameter :: species_names(${sol.n_species}) = &
        (/ ${", ".join('"'+'{0: <12}'.format(s)+'"' for s in sol.species_names)} /)

    character(len=4), parameter :: element_names(${sol.n_elements}) = &
        (/ ${", ".join('"'+'{0: <4}'.format(e)+'"' for e in sol.element_names)} /)
    %endif

contains

    %if multi:
    subroutine set_mechanism(mech_name, status)

        character(len=*), intent(in) :: mech_name
        integer, intent(out) :: status

        status = 0
        select case (trim(adjustl(mech_name)))
        %for m in mechs:
        case ("${m["name"]}")
            mech_id = ${m["id"]}
            num_species = ${m["sol"].n_species}
            num_reactions = ${m["sol"].n_reactions}
            num_elements = ${m["sol"].n_elements}
            num_falloff = ${len(m["falloff"])}
            molecular_weights(1:${m["sol"].n_species}) = &
                (/ ${str_np(m["sol"].molecular_weights)} /)
            inv_molecular_weights(1:${m["sol"].n_species}) = &
                (/ ${str_np(1/m["sol"].molecular_weights)} /)
            species_names(1:${m["sol"].n_species}) = &
                (/ ${", ".join('"'+'{0: <12}'.format(x)+'"' for x in m["sol"].species_names)} /)
            element_names(1:${m["sol"].n_elements}) = &
                (/ ${", ".join('"'+'{0: <4}'.format(x)+'"' for x in m["sol"].element_names)} /)
        %endfor
        case default
            status = -1
        end select

    end subroutine set_mechanism

    %endif
    subroutine get_species_name(sp_index, sp_name)

        integer, intent(in) :: sp_index
        character(len=*), intent(out) :: sp_name

        sp_name = species_names(sp_index + PYROMETHEUS_CALLER_INDEXING)

    end subroutine get_species_name

    subroutine get_species_index(sp_name, sp_index)

        character(len=*), intent(in) :: sp_name
        integer, intent(out) :: sp_index

        integer :: idx

        sp_index = 0
        loop:do idx = 1, num_species
            if(trim(adjustl(sp_name)) .eq. trim(species_names(idx))) then
                sp_index = idx - PYROMETHEUS_CALLER_INDEXING
                exit loop
            end if
        end do loop

    end subroutine get_species_index

    subroutine get_element_index(el_name, el_index)

        character(len=*), intent(in) :: el_name
        integer, intent(out) :: el_index

        integer :: idx

        el_index = 0
        loop:do idx = 1, num_elements
            if(trim(adjustl(el_name)) .eq. trim(element_names(idx))) then
                el_index = idx - PYROMETHEUS_CALLER_INDEXING
                exit loop
            end if
        end do loop

    end subroutine get_element_index

<%def name="_body_get_specific_gas_constant(sol, falloff_reactions, three_body_reactions)">
        specific_gas_constant = gas_constant * ( &
                %for i in range(sol.n_species):
                    + inv_molecular_weights(${i+1})*mass_fractions(${i+1}) &
                %endfor
                )

</%def>
    subroutine get_specific_gas_constant(mass_fractions, specific_gas_constant)

        GPU_ROUTINE(get_specific_gas_constant)

        ${real_type}, intent(in), dimension(${ns_dim}) :: mass_fractions
        ${real_type}, intent(out) :: specific_gas_constant

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_specific_gas_constant(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_specific_gas_constant(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_specific_gas_constant

    subroutine get_density(pressure, temperature, mass_fractions, density)

        GPU_ROUTINE(get_density)

        ${real_type}, intent(in) :: pressure
        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${ns_dim}) :: mass_fractions
        ${real_type}, intent(out) :: density

        ${real_type} :: mix_mol_weight

        call get_mixture_molecular_weight(mass_fractions, mix_mol_weight)
        density = pressure * mix_mol_weight / (gas_constant * temperature)

    end subroutine get_density

    subroutine get_pressure(density, temperature, mass_fractions, pressure)

        GPU_ROUTINE(get_pressure)

        ${real_type}, intent(in) :: density
        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${ns_dim}) :: mass_fractions
        ${real_type}, intent(out) :: pressure

        ${real_type} :: mix_mol_weight

        call get_mixture_molecular_weight(mass_fractions, mix_mol_weight)
        pressure = density * gas_constant * temperature / mix_mol_weight

    end subroutine get_pressure

<%def name="_body_get_mixture_molecular_weight(sol, falloff_reactions, three_body_reactions)">
        mix_mol_weight = 1.0d0 / ( &
                %for i in range(sol.n_species):
                    + inv_molecular_weights(${i+1})*mass_fractions(${i+1}) &
                %endfor
                )

</%def>
    subroutine get_mixture_molecular_weight(mass_fractions, mix_mol_weight)

        GPU_ROUTINE(get_mixture_molecular_weight)

        ${real_type}, intent(in), dimension(${ns_dim}) :: mass_fractions
        ${real_type}, intent(out) :: mix_mol_weight

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_mixture_molecular_weight(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_mixture_molecular_weight(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_mixture_molecular_weight

<%def name="_body_get_concentrations(sol, falloff_reactions, three_body_reactions)">
        %for i in range(sol.n_species):
            concentrations(${i+1}) = density * &
                inv_molecular_weights(${i+1}) * mass_fractions(${i+1})
        %endfor

</%def>
    subroutine get_concentrations(density, mass_fractions, concentrations)

        GPU_ROUTINE(get_concentrations)

        ${real_type}, intent(in) :: density
        ${real_type}, intent(in),  dimension(${ns_dim}) :: mass_fractions
        ${real_type}, intent(out), dimension(${ns_dim}) :: concentrations

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_concentrations(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_concentrations(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_concentrations

<%def name="_body_get_mole_fractions(sol, falloff_reactions, three_body_reactions)">
        %for i in range(sol.n_species):
            mole_fractions(${i+1}) = inv_molecular_weights(${i+1}) * &
                mass_fractions(${i+1}) * mix_mol_weight
        %endfor

</%def>
    subroutine get_mole_fractions(mix_mol_weight, mass_fractions, mole_fractions)

        GPU_ROUTINE(get_mole_fractions)

        ${real_type}, intent(in) :: mix_mol_weight
        ${real_type}, intent(in),  dimension(${ns_dim}) :: mass_fractions
        ${real_type}, intent(out), dimension(${ns_dim}) :: mole_fractions

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_mole_fractions(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_mole_fractions(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_mole_fractions

<%def name="_body_get_mass_averaged_property(sol, falloff_reactions, three_body_reactions)">
        mix_property =  ( &
            %for i in range(sol.n_species):
                + inv_molecular_weights(${i+1})*mass_fractions(${i+1}) &
                *spec_property(${i+1}) &
            %endfor
        )

</%def>
    subroutine get_mass_averaged_property(&
        & mass_fractions, spec_property, mix_property)

        GPU_ROUTINE(get_mass_averaged_property)

        ${real_type}, intent(in), dimension(${ns_dim}) :: mass_fractions
        ${real_type}, intent(in), dimension(${ns_dim}) :: spec_property
        ${real_type}, intent(out) :: mix_property

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_mass_averaged_property(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_mass_averaged_property(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_mass_averaged_property

    subroutine get_mixture_specific_heat_cp_mass(temperature, mass_fractions, cp_mix)

        GPU_ROUTINE(get_mixture_specific_heat_cp_mass)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${ns_dim}) :: mass_fractions
        ${real_type}, intent(out) :: cp_mix

        ${real_type}, dimension(${ns_dim}) :: cp0_r

        call get_species_specific_heats_r(temperature, cp0_r)
        call get_mass_averaged_property(mass_fractions, cp0_r, cp_mix)
        cp_mix = cp_mix * gas_constant

    end subroutine get_mixture_specific_heat_cp_mass

<%def name="_body_get_mixture_specific_heat_cv_mass(sol, falloff_reactions, three_body_reactions)">
        call get_species_specific_heats_r(temperature, cp0_r)

        %for i in range(sol.n_species):
            cp0_r(${i+1}) = cp0_r(${i+1}) - 1.d0
        %endfor

        call get_mass_averaged_property(mass_fractions, cp0_r, cv_mix)
        cv_mix = cv_mix * gas_constant

</%def>
    subroutine get_mixture_specific_heat_cv_mass(temperature, mass_fractions, cv_mix)

        GPU_ROUTINE(get_mixture_specific_heat_cv_mass)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${ns_dim}) :: mass_fractions
        ${real_type}, intent(out) :: cv_mix

        ${real_type}, dimension(${ns_dim}) :: cp0_r

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_mixture_specific_heat_cv_mass(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_mixture_specific_heat_cv_mass(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_mixture_specific_heat_cv_mass

    subroutine get_mixture_enthalpy_mass(temperature, mass_fractions, h_mix)

        GPU_ROUTINE(get_mixture_enthalpy_mass)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${ns_dim}) :: mass_fractions
        ${real_type}, intent(out) :: h_mix

        ${real_type}, dimension(${ns_dim}) :: h0_rt

        call get_species_enthalpies_rt(temperature, h0_rt)
        call get_mass_averaged_property(mass_fractions, h0_rt, h_mix)
        h_mix = h_mix * gas_constant * temperature

    end subroutine get_mixture_enthalpy_mass

<%def name="_body_get_mixture_energy_mass(sol, falloff_reactions, three_body_reactions)">
        call get_species_enthalpies_rt(temperature, h0_rt)

        %for i in range(sol.n_species):
            h0_rt(${i+1}) = h0_rt(${i+1}) - 1.d0
        %endfor

        call get_mass_averaged_property(mass_fractions, h0_rt, e_mix)
        e_mix = e_mix * gas_constant * temperature

</%def>
    subroutine get_mixture_energy_mass(temperature, mass_fractions, e_mix)

        GPU_ROUTINE(get_mixture_energy_mass)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${ns_dim}) :: mass_fractions
        ${real_type}, intent(out) :: e_mix

        ${real_type}, dimension(${ns_dim}) :: h0_rt

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_mixture_energy_mass(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_mixture_energy_mass(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_mixture_energy_mass

<%def name="_body_get_species_specific_heats_r(sol, falloff_reactions, three_body_reactions)">
        %for i, sp in enumerate(sol.species()):
        cp0_r(${i+1}) = ${cgm(ce.poly_to_expr(sp.thermo, "temperature"))}
        %endfor

</%def>
    subroutine get_species_specific_heats_r(temperature, cp0_r)

        GPU_ROUTINE(get_species_specific_heats_r)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(out), dimension(${ns_dim}) :: cp0_r

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_species_specific_heats_r(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_species_specific_heats_r(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_species_specific_heats_r

<%def name="_body_get_species_enthalpies_rt(sol, falloff_reactions, three_body_reactions)">
        %for i, sp in enumerate(sol.species()):
        h0_rt(${i+1}) = ${cgm(ce.poly_to_enthalpy_expr(sp.thermo, "temperature"))}
        %endfor

</%def>
    subroutine get_species_enthalpies_rt(temperature, h0_rt)

        GPU_ROUTINE(get_species_enthalpies_rt)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(out), dimension(${ns_dim}) :: h0_rt

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_species_enthalpies_rt(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_species_enthalpies_rt(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_species_enthalpies_rt

<%def name="_body_get_species_entropies_r(sol, falloff_reactions, three_body_reactions)">
        %for i, sp in enumerate(sol.species()):
        s0_r(${i+1}) = ${cgm(ce.poly_to_entropy_expr(sp.thermo, "temperature"))}
        %endfor

</%def>
    subroutine get_species_entropies_r(temperature, s0_r)

        GPU_ROUTINE(get_species_entropies_r)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(out), dimension(${ns_dim}) :: s0_r

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_species_entropies_r(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_species_entropies_r(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_species_entropies_r

<%def name="_body_get_species_gibbs_rt(sol, falloff_reactions, three_body_reactions)">
        call get_species_enthalpies_rt(temperature, h0_rt)
        call get_species_entropies_r(temperature, s0_r)

        %for i in range(sol.n_species):
            g0_rt(${i+1}) = h0_rt(${i+1}) - s0_r(${i+1})
        %endfor

</%def>
    subroutine get_species_gibbs_rt(temperature, g0_rt)

        GPU_ROUTINE(get_species_gibbs_rt)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(out), dimension(${ns_dim}) :: g0_rt

        ${real_type}, dimension(${ns_dim}) :: h0_rt
        ${real_type}, dimension(${ns_dim}) :: s0_r

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_species_gibbs_rt(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_species_gibbs_rt(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_species_gibbs_rt

<%def name="_body_get_equilibrium_constants(sol, falloff_reactions, three_body_reactions)">
        rt = gas_constant * temperature
        c0 = log(one_atm/rt)

        call get_species_gibbs_rt(temperature, g0_rt)

        %for i, react in enumerate(sol.reactions()):
        %if react.reversible:
        k_eq(${i+1}) = ${cgm(
            ce.equilibrium_constants_expr(sol, i, Variable("g0_rt")))}
        %else:
        k_eq(${i+1}) = -0.1d0*temperature
        %endif
        %endfor

</%def>
    subroutine get_equilibrium_constants(temperature, k_eq)

        GPU_ROUTINE(get_equilibrium_constants)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(out), dimension(${nr_dim}) :: k_eq

        ${real_type} :: rt
        ${real_type} :: c0

        ${real_type}, dimension(${ns_dim}) :: g0_rt

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_equilibrium_constants(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_equilibrium_constants(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_equilibrium_constants

    subroutine get_temperature( &
        & enthalpy_or_energy, t_guess, mass_fractions, do_energy, temperature)

        GPU_ROUTINE(get_temperature)

        logical, intent(in) :: do_energy
        ${real_type}, intent(in)  :: enthalpy_or_energy
        ${real_type}, intent(in)  :: t_guess
        ${real_type}, intent(in), dimension(${ns_dim}) :: mass_fractions
        ${real_type}, intent(out) :: temperature

        integer :: iter
        integer,      parameter :: num_iter = 500
        ${real_type}, parameter :: tol = 1.0d-06

        ${real_type} :: iter_temp
        ${real_type} :: iter_energy
        ${real_type} :: iter_energy_deriv
        ${real_type} :: iter_rhs
        ${real_type} :: iter_deriv

        iter_rhs = 0.d0
        iter_deriv = 1.d0
        iter_temp = t_guess

        do iter = 1, num_iter
            if(do_energy) then
                call get_mixture_specific_heat_cv_mass(&
                    & iter_temp, mass_fractions, iter_energy_deriv)
                call get_mixture_energy_mass(iter_temp, mass_fractions, iter_energy)
            else
                call get_mixture_specific_heat_cp_mass(&
                    & iter_temp, mass_fractions, iter_energy_deriv)
                call get_mixture_enthalpy_mass(&
                    & iter_temp, mass_fractions, iter_energy)
            endif
            iter_rhs = enthalpy_or_energy - iter_energy
            iter_deriv = (-1.d0)*iter_energy_deriv
            iter_temp = iter_temp - iter_rhs / iter_deriv
            if(abs(iter_rhs/iter_deriv) .lt. tol) exit
        end do

        temperature = iter_temp

    end subroutine get_temperature

    %if any_falloff:
<%def name="_body_get_falloff_rates(sol, falloff_reactions, three_body_reactions)">
        %for i, (_, react) in enumerate(falloff_reactions):
        k_high(${i+1}) = ${cgm(ce.rate_coefficient_expr(
                                react.rate.high_rate,
                                Variable("temperature")))}
        %endfor

        %for i, (_, react) in enumerate(falloff_reactions):
        k_low(${i+1}) = ${cgm(ce.rate_coefficient_expr(
                                react.rate.low_rate,
                                Variable("temperature")))}
        %endfor

        %for i, (_, react) in enumerate(falloff_reactions):
        reduced_pressure(${i+1}) = (${cgm(
            ce.third_body_efficiencies_expr(sol,
                                            react,
                                            Variable("concentrations")))})*k_low(${i+1})/k_high(${i+1})
        %endfor

        %for i, (_, react) in enumerate(falloff_reactions):
        falloff_center(${i+1}) = ${cgm(ce.troe_falloff_center_expr(
            react, Variable("temperature")))}
        %endfor

        %for i, (_, react) in enumerate(falloff_reactions):
        falloff_factor(${i+1}) = ${cgm(ce.troe_falloff_factor_expr(react, i,
            Variable("reduced_pressure"), Variable("falloff_center")))}
        %endfor

        %for i, (_, react) in enumerate(falloff_reactions):
        falloff_function(${i+1}) = ${cgm(ce.falloff_function_expr(
            react, i,
            Variable("falloff_factor"),
            Variable("falloff_center")))}
        %endfor

        %for i, (j, react) in enumerate(falloff_reactions):
        k_fwd(${j+1}) = k_high(${i+1})*falloff_function(${i+1}) * &
            reduced_pressure(${i+1})/(1.d0 + reduced_pressure(${i+1}))
        %endfor

</%def>
    subroutine get_falloff_rates(temperature, concentrations, k_fwd)

        GPU_ROUTINE(get_falloff_rates)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${ns_dim}) :: concentrations
        ${real_type}, intent(out), dimension(${nr_dim}) :: k_fwd

        ${real_type}, dimension(${nf_dim}) :: k_high
        ${real_type}, dimension(${nf_dim}) :: k_low
        ${real_type}, dimension(${nf_dim}) :: reduced_pressure
        ${real_type}, dimension(${nf_dim}) :: falloff_center
        ${real_type}, dimension(${nf_dim}) :: falloff_factor
        ${real_type}, dimension(${nf_dim}) :: falloff_function

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_falloff_rates(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_falloff_rates(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_falloff_rates

    %endif
<%def name="_body_get_fwd_rate_coefficients(sol, falloff_reactions, three_body_reactions)">
        %for i, react in enumerate(sol.reactions()):
        %if react.equation in [r.equation for _, r in falloff_reactions]:
        k_fwd(${i+1}) = 0.d0
        %else:
        k_fwd(${i+1}) = ${cgm(ce.rate_coefficient_expr(react.rate,
                            Variable("temperature")))}
        %endif
        %endfor

        %for j, react in three_body_reactions:
        k_fwd(${j+1}) = k_fwd(${j+1}) * ( &
            ${cgm(ce.third_body_efficiencies_expr(
            sol, react, Variable("concentrations")))})
        %endfor

        %if falloff_reactions:
        call get_falloff_rates(temperature, concentrations, k_fwd)
        %endif

</%def>
    subroutine get_fwd_rate_coefficients(temperature, concentrations, k_fwd)

        GPU_ROUTINE(get_fwd_rate_coefficients)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${ns_dim}) :: concentrations
        ${real_type}, intent(out), dimension(${nr_dim}) :: k_fwd

        %if falloff_reactions:
        ${real_type}, dimension(${nf_dim}) :: k_falloff
        %endif

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_fwd_rate_coefficients(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_fwd_rate_coefficients(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_fwd_rate_coefficients

<%def name="_body_get_net_rates_of_progress(sol, falloff_reactions, three_body_reactions)">
        call get_fwd_rate_coefficients(temperature, concentrations, k_fwd)
        call get_equilibrium_constants(temperature, log_k_eq)
        %for i in range(sol.n_reactions):
        r_net(${i+1}) = ${cgm(ce.rate_of_progress_expr(sol, i,
                        Variable("concentrations"),
                        Variable("k_fwd"), Variable("log_k_eq")))}
        %endfor

</%def>
    subroutine get_net_rates_of_progress(temperature, concentrations, r_net)

        GPU_ROUTINE(get_net_rates_of_progress)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${ns_dim}) :: concentrations
        ${real_type}, intent(out), dimension(${nr_dim}) :: r_net

        ${real_type}, dimension(${nr_dim}) :: k_fwd
        ${real_type}, dimension(${nr_dim}) :: log_k_eq

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_net_rates_of_progress(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_net_rates_of_progress(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_net_rates_of_progress

<%def name="_body_get_net_production_rates(sol, falloff_reactions, three_body_reactions)">
        call get_concentrations(density, mass_fractions, concentrations)
        call get_net_rates_of_progress(temperature, concentrations, r_net)

        %for i, sp in enumerate(sol.species()):
        omega(${i+1}) = ${cgm(ce.production_rate_expr(sol,
            sp.name, Variable("r_net")))}
        %endfor

</%def>
    subroutine get_net_production_rates(density, temperature, mass_fractions, omega)

        GPU_ROUTINE(get_net_production_rates)

        ${real_type}, intent(in) :: density
        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in),  dimension(${ns_dim}) :: mass_fractions
        ${real_type}, intent(out), dimension(${ns_dim}) :: omega

        ${real_type}, dimension(${ns_dim})   :: concentrations
        ${real_type}, dimension(${nr_dim}) :: r_net

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_net_production_rates(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_net_production_rates(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_net_production_rates

<%def name="_body_get_fwd_rates_of_progress(sol, falloff_reactions, three_body_reactions)">
        call get_fwd_rate_coefficients(temperature, concentrations, k_fwd)
        %for i in range(sol.n_reactions):
        r_fwd(${i+1}) = ${cgm(ce.fwd_rate_of_progress_expr(sol, i,
                        Variable("concentrations"), Variable("k_fwd")))}
        %endfor

</%def>
    subroutine get_fwd_rates_of_progress(temperature, concentrations, r_fwd)

        GPU_ROUTINE(get_fwd_rates_of_progress)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${ns_dim}) :: concentrations
        ${real_type}, intent(out), dimension(${nr_dim}) :: r_fwd

        ${real_type}, dimension(${nr_dim}) :: k_fwd

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_fwd_rates_of_progress(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_fwd_rates_of_progress(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_fwd_rates_of_progress

<%def name="_body_get_rev_rates_of_progress(sol, falloff_reactions, three_body_reactions)">
        call get_fwd_rate_coefficients(temperature, concentrations, k_fwd)
        call get_equilibrium_constants(temperature, log_k_eq)
        %for i in range(sol.n_reactions):
        r_rev(${i+1}) = ${cgm(ce.rev_rate_of_progress_expr(sol, i,
                        Variable("concentrations"),
                        Variable("k_fwd"), Variable("log_k_eq")))}
        %endfor

</%def>
    subroutine get_rev_rates_of_progress(temperature, concentrations, r_rev)

        GPU_ROUTINE(get_rev_rates_of_progress)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${ns_dim}) :: concentrations
        ${real_type}, intent(out), dimension(${nr_dim}) :: r_rev

        ${real_type}, dimension(${nr_dim}) :: k_fwd
        ${real_type}, dimension(${nr_dim}) :: log_k_eq

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_rev_rates_of_progress(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_rev_rates_of_progress(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_rev_rates_of_progress

<%def name="_body_get_creation_rates(sol, falloff_reactions, three_body_reactions)">
        call get_concentrations(density, mass_fractions, concentrations)
        call get_fwd_rates_of_progress(temperature, concentrations, r_fwd)
        call get_rev_rates_of_progress(temperature, concentrations, r_rev)

        %for i, sp in enumerate(sol.species()):
        cdot(${i+1}) = ${cgm(ce.creation_rate_expr(sol, sp.name,
            Variable("r_fwd"), Variable("r_rev")))}
        %endfor

</%def>
    subroutine get_creation_rates(density, temperature, mass_fractions, cdot)

        GPU_ROUTINE(get_creation_rates)

        ${real_type}, intent(in) :: density
        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in),  dimension(${ns_dim}) :: mass_fractions
        ${real_type}, intent(out), dimension(${ns_dim}) :: cdot

        ${real_type}, dimension(${ns_dim})   :: concentrations
        ${real_type}, dimension(${nr_dim}) :: r_fwd, r_rev

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_creation_rates(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_creation_rates(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_creation_rates

<%def name="_body_get_destruction_rates(sol, falloff_reactions, three_body_reactions)">
        call get_concentrations(density, mass_fractions, concentrations)
        call get_fwd_rates_of_progress(temperature, concentrations, r_fwd)
        call get_rev_rates_of_progress(temperature, concentrations, r_rev)

        %for i, sp in enumerate(sol.species()):
        ddot(${i+1}) = ${cgm(ce.destruction_rate_expr(sol, sp.name,
            Variable("r_fwd"), Variable("r_rev")))}
        %endfor

</%def>
    subroutine get_destruction_rates(density, temperature, mass_fractions, ddot)

        GPU_ROUTINE(get_destruction_rates)

        ${real_type}, intent(in) :: density
        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in),  dimension(${ns_dim}) :: mass_fractions
        ${real_type}, intent(out), dimension(${ns_dim}) :: ddot

        ${real_type}, dimension(${ns_dim})   :: concentrations
        ${real_type}, dimension(${nr_dim}) :: r_fwd, r_rev

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_destruction_rates(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_destruction_rates(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_destruction_rates

<%def name="_body_get_creation_destruction_rates(sol, falloff_reactions, three_body_reactions)">
        call get_concentrations(density, mass_fractions, concentrations)
        call get_fwd_rates_of_progress(temperature, concentrations, r_fwd)
        call get_rev_rates_of_progress(temperature, concentrations, r_rev)

        %for i, sp in enumerate(sol.species()):
        cdot(${i+1}) = ${cgm(ce.creation_rate_expr(sol, sp.name,
            Variable("r_fwd"), Variable("r_rev")))}
        ddot(${i+1}) = ${cgm(ce.destruction_rate_expr(sol, sp.name,
            Variable("r_fwd"), Variable("r_rev")))}
        %endfor

</%def>
    subroutine get_creation_destruction_rates(density, temperature, &
        mass_fractions, cdot, ddot)

        GPU_ROUTINE(get_creation_destruction_rates)

        ${real_type}, intent(in) :: density
        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in),  dimension(${ns_dim}) :: mass_fractions
        ${real_type}, intent(out), dimension(${ns_dim}) :: cdot, ddot

        ${real_type}, dimension(${ns_dim})   :: concentrations
        ${real_type}, dimension(${nr_dim}) :: r_fwd, r_rev

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_creation_destruction_rates(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_creation_destruction_rates(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_creation_destruction_rates

<%def name="_body_get_species_viscosities(sol, falloff_reactions, three_body_reactions)">
        %for sp in range(sol.n_species):
        viscosities(${sp+1}) = ${cgm(ce.viscosity_polynomial_expr(
            sol.get_viscosity_polynomial(sp),
            Variable("temperature")))}
        %endfor

</%def>
    subroutine get_species_viscosities(temperature, viscosities)

        GPU_ROUTINE(get_species_viscosities)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(out), dimension(${ns_dim}) :: viscosities

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_species_viscosities(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_species_viscosities(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_species_viscosities

<%def name="_body_get_species_thermal_conductivities(sol, falloff_reactions, three_body_reactions)">
        %for sp in range(sol.n_species):
        conductivities(${sp+1}) = ${cgm(ce.conductivity_polynomial_expr(
            sol.get_thermal_conductivity_polynomial(sp),
            Variable("temperature")))}
        %endfor

</%def>
    subroutine get_species_thermal_conductivities(temperature, conductivities)

        GPU_ROUTINE(get_species_thermal_conductivities)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(out), dimension(${ns_dim}) :: conductivities

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_species_thermal_conductivities(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_species_thermal_conductivities(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_species_thermal_conductivities

<%def name="_body_get_species_binary_mass_diffusivities(sol, falloff_reactions, three_body_reactions)">
        %for i in range(sol.n_species):
        %for j in range(sol.n_species):
        diffusivities(${i + 1}, ${j + 1}) = ${cgm(ce.diffusivity_polynomial_expr(
            sol.get_binary_diff_coeffs_polynomial(i, j),
            Variable("temperature")))}
        %endfor
        %endfor

</%def>
    subroutine get_species_binary_mass_diffusivities(temperature, diffusivities)

        GPU_ROUTINE(get_species_binary_mass_diffusivities)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(out), dimension(${ns_dim}, ${ns_dim})&
            :: diffusivities

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_species_binary_mass_diffusivities(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_species_binary_mass_diffusivities(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_species_binary_mass_diffusivities

<%def name="_body_get_mixture_viscosity_mixavg(sol, falloff_reactions, three_body_reactions)">
        call get_mixture_molecular_weight(mass_fractions, mix_mol_weight)
        call get_mole_fractions(mix_mol_weight, mass_fractions, mole_fractions)
        call get_species_viscosities(temperature, viscosities)

        %for sp in range(sol.n_species):
        mix_rule_f(${sp + 1}) = ${cgm(ce.viscosity_mixture_rule_wilke_expr(sol, sp,
            Variable("mole_fractions"), Variable("viscosities")))}
        %endfor

        mixture_viscosity_mixavg = sum(mole_fractions*viscosities/mix_rule_f)

</%def>
    subroutine get_mixture_viscosity_mixavg(&
        temperature, mass_fractions, mixture_viscosity_mixavg)

        GPU_ROUTINE(get_mixture_viscosity_mixavg)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${ns_dim}) :: mass_fractions
        ${real_type}, intent(out) :: mixture_viscosity_mixavg

        ${real_type} :: mix_mol_weight
        ${real_type}, dimension(${ns_dim}) :: &
            mole_fractions, viscosities, mix_rule_f

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_mixture_viscosity_mixavg(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_mixture_viscosity_mixavg(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_mixture_viscosity_mixavg

    subroutine get_mixture_thermal_conductivity_mixavg(temperature, &
        mass_fractions, mixture_thermal_conductivity_mixavg)

        GPU_ROUTINE(get_mixture_thermal_conductivity_mixavg)

        ${real_type}, intent(in) :: temperature
        ${real_type}, intent(in), dimension(${ns_dim}) :: mass_fractions
        ${real_type}, intent(out) :: mixture_thermal_conductivity_mixavg

        ${real_type} :: mix_mol_weight
        ${real_type}, dimension(${ns_dim}) :: mole_fractions, conductivities

        call get_mixture_molecular_weight(mass_fractions, mix_mol_weight)
        call get_mole_fractions(mix_mol_weight, mass_fractions, mole_fractions)
        call get_species_thermal_conductivities(temperature, conductivities)

        mixture_thermal_conductivity_mixavg = 0.5*(&
            sum(mole_fractions*conductivities) + &
            1/sum(mole_fractions/conductivities))

    end subroutine get_mixture_thermal_conductivity_mixavg

<%def name="_body_get_species_mass_diffusivities_mixavg(sol, falloff_reactions, three_body_reactions)">
        call get_mixture_molecular_weight(mass_fractions, mix_mol_weight)
        call get_mole_fractions(mix_mol_weight, mass_fractions, mole_fractions)
        call get_species_binary_mass_diffusivities(temperature, bdiff_ij)

        %for sp in range(sol.n_species):
        x_sum(${sp + 1}) = ${cgm(ce.diffusivity_mixture_rule_denom_expr(
                sol, sp, Variable("mole_fractions"), Variable("bdiff_ij")))}
        %endfor

        %for sp in range(sol.n_species):
        denom(${sp + 1}) = x_sum(${sp + 1}) - &
            mole_fractions(${sp + 1})/bdiff_ij(${sp + 1}, ${sp + 1})
        %endfor

        %for sp in range(sol.n_species):
        if (denom(${sp + 1}) .gt. 0d0) then
        mass_diffusivities_mixavg(${sp + 1}) = &
            (mix_mol_weight - &
                mole_fractions(${sp + 1})*molecular_weights(${sp + 1}))&
            /(pressure * mix_mol_weight * denom(${sp + 1}))
        else
        mass_diffusivities_mixavg(${sp + 1}) = &
            bdiff_ij(${sp + 1}, ${sp + 1}) / pressure
        end if
        %endfor

</%def>
    subroutine get_species_mass_diffusivities_mixavg(&
        pressure, temperature, mass_fractions, mass_diffusivities_mixavg)

        GPU_ROUTINE(get_species_mass_diffusivities_mixavg)

        ${real_type}, intent(in) :: pressure, temperature
        ${real_type}, intent(in), dimension(${ns_dim}) :: mass_fractions
        ${real_type}, intent(out), dimension(${ns_dim}) :: &
            mass_diffusivities_mixavg

        ${real_type} :: mix_mol_weight
        ${real_type}, dimension(${ns_dim}) :: mole_fractions, x_sum, denom
        ${real_type}, dimension(${ns_dim}, ${ns_dim}) :: bdiff_ij

%if multi:
        select case (mech_id)
%for _m in mechs:
        case (${_m["id"]})
${_body_get_species_mass_diffusivities_mixavg(_m['sol'], _m['falloff'], _m['three_body'])}
%endfor
        end select
%else:
${_body_get_species_mass_diffusivities_mixavg(sol, falloff_reactions, three_body_reactions)}
%endif
    end subroutine get_species_mass_diffusivities_mixavg

end module ${module_name}
""")

# }}}


class FortranCodeGenerator(CodeGenerator):
    @staticmethod
    def get_name() -> str:
        return "fortran"

    @staticmethod
    def supports_overloading() -> bool:
        return False

    @staticmethod
    def generate(name: str,
                 sol,
                 opts: CodeGenerationOptions = None) -> str:
        """`sol` is a Cantera Solution, or a non-empty list of them. A list of
        one emits exactly what a bare Solution emits."""
        if opts is None:
            opts = CodeGenerationOptions()

        sols = list(sol) if isinstance(sol, (list, tuple)) else [sol]
        if not sols:
            raise ValueError("at least one mechanism is required")

        mechs = []
        for idx, s in enumerate(sols):
            mechs.append({
                "id": idx + 1,
                "sol": s,
                # NOT s.name -- that is the *phase* name, typically "gas" for
                # every mechanism, which would make selection ambiguous. The
                # source file's stem is the identity a user actually types.
                "name": os.path.splitext(os.path.basename(s.source))[0],
                "falloff": [(i, r) for i, r in enumerate(s.reactions())
                            if r.reaction_type.startswith("falloff")],
                "three_body": [(i, r) for i, r in enumerate(s.reactions())
                               if r.reaction_type == "three-body-Arrhenius"],
            })
        multi = len(mechs) > 1
        num_species_max = max(m["sol"].n_species for m in mechs)
        num_reactions_max = max(m["sol"].n_reactions for m in mechs)
        num_elements_max = max(m["sol"].n_elements for m in mechs)
        sol = sols[0]

        if opts.directive_offload == "acc":
            gpu_routine_str = """
#define GPU_ROUTINE(name) !$acc routine seq
"""
        elif opts.directive_offload == "mp":
            gpu_routine_str = """
#define GPU_ROUTINE(name) !$omp declare target
"""
        else:
            gpu_routine_str = """
#define GPU_ROUTINE(name) ! name
"""

        falloff_rxn = [(i, r) for i, r in enumerate(sol.reactions())
                    if r.reaction_type.startswith("falloff")]
        three_body_rxn = [(i, r) for i, r in enumerate(sol.reactions())
                        if r.reaction_type == "three-body-Arrhenius"]

        return wrap_code(module_tpl.render(
            ct=ct,
            sol=sol,

            str_np=str_np,
            cgm=FortranExpressionMapper(),
            Variable=p.Variable,
            float_to_fortran=float_to_fortran,

            real_type=opts.scalar_type or "real(dp)",
            gpu_routine=gpu_routine_str,

            module_name=name,

            ce=pyrometheus.chem_expr,

            falloff_reactions=falloff_rxn,
            three_body_reactions=three_body_rxn,

            mechs=mechs,
            multi=multi,
            ns_dim=("num_species_max" if multi else str(sol.n_species)),
            nr_dim=("num_reactions_max" if multi else str(sol.n_reactions)),
            num_species_max=num_species_max,
            num_reactions_max=num_reactions_max,
            num_elements_max=num_elements_max,
            nf_dim=(str(max(len(m["falloff"]) for m in mechs))
                    if multi else str(len(falloff_rxn))),
            any_falloff=(any(m["falloff"] for m in mechs)
                         if multi else bool(falloff_rxn))
        ))


# vim: foldmethod=marker
