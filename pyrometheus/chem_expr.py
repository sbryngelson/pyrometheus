__copyright__ = """
Copyright (C) 2020 Esteban Cisneros
Copyright (C) 2020 Andreas Kloeckner
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

"""
Internal Functionality
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: equilibrium_constants_expr
.. autofunction:: rate_coefficient_expr
.. autofunction:: third_body_efficiencies_expr
.. autofunction:: troe_falloff_expr
.. autofunction:: falloff_function_expr
.. autofunction:: rate_of_progress_expr
.. autofunction:: production_rate_expr
"""

import pymbolic.primitives as p
from pymbolic import mapper
from functools import singledispatch
import cantera as ct
import numpy as np


# {{{ polynomial processing

def nasa7_conditional(variables, poly, part_gen):
    # FIXME: Should check minTemp, maxTemp
    return p.If(
        p.Comparison(variables[0], ">", poly.coeffs[0]),
        part_gen(poly.coeffs[1:8], *variables),
        part_gen(poly.coeffs[8:15], *variables),
    )


@singledispatch
def poly_to_expr(poly):
    raise TypeError(f"unexpected argument type in poly_to_expr: {type(poly)}")


@poly_to_expr.register
def _(poly: ct.NasaPoly2, arg_names):
    variables = [p.Variable(arg) for arg in arg_names]
    def gen(c, t, ov_t, t2, t3, t4, log_t):
        assert len(c) == 7
        return (
            c[0]
            + c[1] * t
            + c[2] * t2
            + c[3] * t3
            + c[4] * t4
        )

    return nasa7_conditional(variables, poly, gen)


@singledispatch
def poly_to_enthalpy_expr(poly, arg_names):
    raise TypeError("unexpected argument type in poly_to_enthalpy_expr: "
            f"{type(poly)}")


@poly_to_enthalpy_expr.register
def _(poly: ct.NasaPoly2, arg_names):
    variables = [p.Variable(arg) for arg in arg_names]
    def gen(c, t, ov_t, t2, t3, t4, log_t):
        assert len(c) == 7
        return (
            c[0]
            + c[1] / 2 * t
            + c[2] / 3 * t2
            + c[3] / 4 * t3
            + c[4] / 5 * t4
            + c[5] * ov_t
        )

    return nasa7_conditional(variables, poly, gen)


@singledispatch
def poly_to_entropy_expr(poly, arg_name):
    raise TypeError("unexpected argument type in poly_to_entropy_expr: "
            f"{type(poly)}")


@poly_to_entropy_expr.register
def _(poly: ct.NasaPoly2, arg_names):
    #log = p.Variable("log")
    variables = [p.Variable(arg) for arg in arg_names]

    def gen(c, t, ov_t, t2, t3, t4, log_t):
        assert len(c) == 7
        return (
            c[0] * log_t
            + c[1] * t
            + c[2] / 2 * t2
            + c[3] / 3 * t3
            + c[4] / 4 * t4
            + c[6]
        )

    return nasa7_conditional(variables, poly, gen)


@singledispatch
def poly_to_enthalpy_deriv_expr(poly, arg_name):
    raise TypeError("unexpected argument type in poly_to_enthalpy_deriv_expr: "
            f"{type(poly)}")


@poly_to_enthalpy_deriv_expr.register
def _(poly: ct.NasaPoly2, arg_names):
    variables = [p.Variable(arg) for arg in arg_names]
    def gen(c, t, ov_t, t2, t3, t4, log_t):
        assert len(c) == 7
        return (
            c[1] / 2
            + 2 * c[2] / 3 * t
            + 3 * c[3] / 4 * t2
            + 4 * c[4] / 5 * t3
            - c[5] / (t2)
        )

    return nasa7_conditional(variables, poly, gen)


@singledispatch
def poly_to_entropy_deriv_expr(poly, arg_name):
    raise TypeError("unexpected argument type in poly_to_entropy_deriv_expr: "
            f"{type(poly)}")


@poly_to_entropy_deriv_expr.register
def _(poly: ct.NasaPoly2, arg_names):
    variables = [p.Variable(arg) for arg in arg_names]
    def gen(c, t, ov_t, t2, t3, t4, log_t):
        assert len(c) == 7
        return (
            c[0] * ov_t
            + c[1]
            + c[2] * t
            + c[3] * t2
            + c[4] * t3
        )

    return nasa7_conditional(variables, poly, gen)

# }}}


# {{{ Data-handling helper

def _zeros_like(argument):
    # FIXME: This mishandles NaNs.
    return 0 * argument

# }}}

# {{{ Equilibrium constants


def equilibrium_constants_expr(sol: ct.Solution, reaction_index, gibbs_rt):
    """Generate code for equilibrium constants.

    :returns: Equilibrium constant expression for reaction with
        index *reaction_index* in terms of the species Gibbs
        functions *gibbs_rt* as a :class:`pymbolic.primitives.Expression`
    """

    indices_reac = [sol.species_index(sp)
                    for sp in sol.reaction(reaction_index).reactants]
    indices_prod = [sol.species_index(sp)
                    for sp in sol.reaction(reaction_index).products]

    # Stoichiometric coefficients
    #nu_reac = [react.reactants[sp] for sp in react.reactants]
    #nu_prod = [react.products[sp] for sp in react.products]
    nu_reac = [sol.reactant_stoich_coeff(sol.species_index(sp), reaction_index)
               for sp in sol.reaction(reaction_index).reactants]
    nu_prod = [sol.product_stoich_coeff(sol.species_index(sp), reaction_index)
               for sp in sol.reaction(reaction_index).products]

    sum_r = sum(nu_reac_i * gibbs_rt[indices_reac_i]
            for indices_reac_i, nu_reac_i in zip(indices_reac, nu_reac))
    sum_p = sum(nu_prod_i * gibbs_rt[indices_prod_i]
            for indices_prod_i, nu_prod_i in zip(indices_prod, nu_prod))

    # Check if reaction is termolecular
    sum_nu_net = sum(nu_prod) - sum(nu_reac)
    if sum_nu_net != 0:
        return sum_p - sum_r - sum_nu_net*p.Variable("c0")
    else:
        return sum_p - sum_r


# }}}


# {{{ Rate coefficients

class ArrheniusExpression(p.Expression):

    def __init__(self, a: p.Variable, b: p.Variable, t_act: p.Variable,
                 t: p.Variable, log_t: p.Variable):
        j = p.Variable("j")
        self.children = (p.subscript(a, j),
                         p.subscript(b, j),
                         p.Subscript(t_act, j), t, log_t)

    def __getinitargs__(self):
        return self.children

    mapper_method = "map_arrhenius"


class ArrheniusMapper(mapper.IdentityMapper):

    def _pre_exponential(self, expr):
        return expr.children[0]

    def _temp_exponent(self, expr):
        return expr.children[1]

    def _activation_temp(self, expr):
        return expr.children[2]

    def _temperature(self, expr):
        return expr.children[3]

    def _log_temperature(self, expr):
        return expr.children[4]

    def _map_arrhenius(self, expr):
        log = p.Variable("log")
        exp = p.Variable("exp")
        t = self._temperature(expr)
        log_t = self._log_temperature(expr)
        return exp(log(self._pre_exponential(expr))
                   + self._temp_exponent(expr) * log_t
                   - self._activation_temp(expr) / t)

    def map_arrhenius(self, expr, rxn_index):
        from pymbolic import substitute
        return substitute(
            self._map_arrhenius(expr), {p.Variable("j"): rxn_index}
        )

    def fixed_coeffs(self, expr, params: ct.Arrhenius):
        from pymbolic import substitute
        a = self._pre_exponential(expr)
        b = self._temp_exponent(expr)
        t_act = self._activation_temp(expr)
        return substitute(self._map_arrhenius(expr), {
            a: params.pre_exponential_factor,
            b: params.temperature_exponent,
            t_act: params.activation_energy/ct.gas_constant
        })


def rate_coefficient_expr(reaction_index, rate_coeff: ct.Arrhenius, a, t):
    """
    :returns: The rate coefficient expression for *rate_coeff* in terms
        of the temperature *t* as a :class:`pymbolic.primitives.Expression`
    """
    # Rate parameters
    # a = rate_coeff.pre_exponential_factor
    b = rate_coeff.temperature_exponent
    t_a = rate_coeff.activation_energy/ct.gas_constant
    if t_a == 0:
        # Weakly temperature-dependent rate
        return a[reaction_index] * t**b
    else:
        # Modified Arrhenius
        return p.Variable("exp")(p.Variable("log")(a[reaction_index])
                                 + b*p.Variable("log")(t)-t_a/t)


def third_body_efficiencies_expr(sol: ct.Solution, react: ct.Reaction, c):
    """
    :returns: The third-body concentration expression for reaction *react* in terms
        of the species concentrations *c* as a
        :class:`pymbolic.primitives.Expression`
    """
    if isinstance(react, ct.ThreeBodyReaction):
        from warnings import warn
        warn("The 'Reaction.efficiencies' interface is deprecated and "
             "will be removed after Cantera 3. Access efficiencies via "
             "'react.third_body.efficiencies' instead")
        efficiencies = [react.efficiencies[sp] for sp in react.efficiencies]
        indices_nondef = [sol.species_index(sp) for sp in react.efficiencies]
        indices_default = [i for i in range(sol.n_species)
                           if i not in indices_nondef]
    else:
        efficiencies = [react.third_body.efficiencies[sp]
                        for sp in react.third_body.efficiencies]
        indices_nondef = [sol.species_index(sp) for sp
                          in react.third_body.efficiencies]
        indices_default = [i for i in range(sol.n_species)
                           if i not in indices_nondef]

    sum_nondef = sum(eff_i * c[index_i] for eff_i, index_i
                     in zip(np.array(efficiencies), indices_nondef))
    sum_default = react.default_efficiency * sum(c[i] for i in indices_default)
    return sum_nondef + sum_default


def troe_falloff_expr(react: ct.Reaction, t):
    """
    :returns: The Troe falloff center expression for reaction *react* in terms of the
        temperature *t* as a :class:`pymbolic.primitives.Expression`
    """
    if "uses_legacy" not in dir(react) or not react.uses_legacy:
        if isinstance(react.rate, ct.TroeRate):
            troe_params = react.rate.falloff_coeffs
        elif isinstance(react.rate, ct.LindemannRate):
            return 1
        else:
            raise ValueError("Unexpected value of 'rate.type': "
                             f" '{react.rate.type}'")
    else:
        from warnings import warn
        warn("Legacy 'ct.Reaction.falloff' interface is deprecated "
             "in Cantera 2.6 and will be removed in Cantera 3. "
             "Access 'FalloffRate' objects using "
             " ct.Reaction.rate' instead", DeprecationWarning, stacklevel=2)
        if react.falloff.falloff_type == "Troe":
            if react.falloff.parameters[3]:
                troe_params = react.falloff.parameters
            else:
                troe_params = react.falloff.parameters[:-1]

        elif react.falloff.falloff_type == "Lindemann":
            return 1
        else:
            raise ValueError("Unexpected value of 'falloff_type': "
                             f" '{react.falloff.falloff_type}'")

    troe_1 = (1.0-troe_params[0])*p.Variable("exp")(-t/troe_params[1])
    troe_2 = troe_params[0]*p.Variable("exp")(-t/troe_params[2])
    if len(troe_params) == 3:
        return p.Variable("log10")(troe_1 + troe_2)
    elif len(troe_params) == 4:
        troe_3 = p.Variable("exp")(-troe_params[3]/t)
        return p.Variable("log10")(troe_1 + troe_2 + troe_3)
    else:
        raise ValueError("Unexpected length of 'tro_params': "
                         f" '{len(troe_params)}'")
    return


def falloff_function_expr(react: ct.Reaction, i, t, red_pressure, falloff_center):
    """
    :returns: Falloff function expression for reaction *react* in terms
        of the temperature *t*, reduced pressure *red_pressure*, and falloff center
        *falloff_center* as a :class:`pymbolic.primitives.Expression`
    """
    if "uses_legacy" not in dir(react) or not react.uses_legacy:
        falloff_type = react.reaction_type.split("-")[1]
    else:
        from warnings import warn
        warn("Legacy 'ct.Reaction.falloff' interface is deprecated "
             "in Cantera 2.6 and will be removed in Cantera 3. "
             "Access 'FalloffRate' objects using "
             " ct.Reaction.rate' instead", DeprecationWarning, stacklevel=2)
        falloff_type = react.falloff.falloff_type

    if falloff_type == "Troe":
        log_rp = p.Variable("log10")(red_pressure[i])
        c = -0.4-0.67*falloff_center[i]
        n = 0.75-1.27*falloff_center[i]
        f = (log_rp+c)/(n-0.14*(log_rp+c))
        return 10**((falloff_center[i])/(1+f**2))
    elif falloff_type == "Lindemann":
        return 1
    else:
        raise ValueError("Unexpected value of 'falloff_type': "
                         f" '{falloff_type}'")

# }}}


# {{{ Rates of progress

def rate_of_progress_expr(sol: ct.Solution, reaction_index, c,
                          k_fwd, log_k_eq):
    """
    :returns: Rate of progress expression for reaction with index *reaction_index*
        in terms of species concentrations *c* with rate coefficients *k_fwd*
        and equilbrium constants *k_eq* as a :class:`pymbolic.primitives.Expression`
    """
    indices_reac = [sol.species_index(sp)
                    for sp in sol.reaction(reaction_index).reactants]
    indices_prod = [sol.species_index(sp)
                    for sp in sol.reaction(reaction_index).products]

    if sol.reaction(reaction_index).orders:
        nu_reac = [sol.reaction(reaction_index).orders[sp]
                   for sp in sol.reaction(reaction_index).orders]
    else:
        nu_reac = [sol.reaction(reaction_index).reactants[sp]
                   for sp in sol.reaction(reaction_index).reactants]

    r_fwd = np.prod([c[index]**nu for index, nu in zip(indices_reac, nu_reac)])

    if sol.reaction(reaction_index).reversible:
        nu_prod = [sol.reaction(reaction_index).products[sp]
                   for sp in sol.reaction(reaction_index).products]
        r_rev = np.prod([c[index]**nu for index, nu in zip(indices_prod, nu_prod)])
        return k_fwd[reaction_index] * (
                r_fwd
                - p.Variable("exp")(log_k_eq[reaction_index]) * r_rev)
    else:
        return k_fwd[reaction_index] * r_fwd

# }}}


# {{{ Species production rates

def production_rate_expr(sol: ct.Solution, species, r_net):
    """
    :returns: Species production rate for species *species* in terms of
        the net reaction rates of progress *r_net* as a
        :class:`pymbolic.primitives.Expression`
    """
    ones = _zeros_like(r_net[0]) + 1.0
    indices_fwd = [i for i, react in enumerate(sol.reactions())
                   if species in react.reactants]
    indices_rev = [i for i, react in enumerate(sol.reactions())
                   if species in react.products]
    nu_fwd = [sol.reactant_stoich_coeff(sol.species_index(species), react_index)
              for react_index in indices_fwd]
    nu_rev = [sol.product_stoich_coeff(sol.species_index(species), prod_index)
              for prod_index in indices_rev]
    sum_fwd = sum(nu*r_net[index] for nu, index in zip(nu_fwd, indices_fwd))
    sum_rev = sum(nu*r_net[index] for nu, index in zip(nu_rev, indices_rev))
    return (sum_rev - sum_fwd) * ones

# }}}

# vim
