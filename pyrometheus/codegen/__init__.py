from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import cantera as ct


@dataclass(frozen=True)
class CodeGenerationOptions:
    scalar_type: Optional[str] = None
    directive_offload: Optional[str] = None
    #: Emit the runtime-selectable form (``num_species`` a variable, bounds on
    #: ``num_species_max``, bodies branching on ``mech_id``) even for a single
    #: mechanism. Without it a single mechanism emits the long-standing
    #: fully-specialized form, byte for byte.
    runtime_mechanism: bool = False


class CodeGenerator:
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Returns the name (slug) of the code generator."""
        pass

    @staticmethod
    @abstractmethod
    def supports_overloading() -> bool:
        """Returns whether the code generator supports operator overloading."""
        pass

    @staticmethod
    @abstractmethod
    def generate(name: str, sol: ct.Solution,
                 opts: CodeGenerationOptions = None) -> str:
        """Invokes Pyrometheus to generate the thermochemistry code for this
        generator and mechanism contained in the passed Cantera Solution object.

        Parameters
        ----------
        name : str
            A module, class, or namespace name for the generated code.
        sol : ct.Solution
            The Cantera Solution object containing the mechanism to generate
            thermochemistry code for.
        opts : CodeGenerationOptions
            Options to pass to the code generator.
        """
        pass
