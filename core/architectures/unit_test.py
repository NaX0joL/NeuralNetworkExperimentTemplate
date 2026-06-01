from core.modules.colored_print import print_colored, print_alternate_color

from .registry import ARCHITECTURE_REGISTRY
from .architecture_control import ArchitectureControl
from .architecture_config import ArchitectureConfig



def architecture_unit_test() -> None:
    for name in ARCHITECTURE_REGISTRY.keys():
        print_alternate_color(f"testing model {name}".upper())
        test_architecture(name)
    return


def test_architecture(architecture_name:str) -> None:
    architecture_config = ArchitectureConfig(
        architecture_name = architecture_name,
        arch_specific_config= ARCHITECTURE_REGISTRY[architecture_name].config_class.default()
    )
    model = ArchitectureControl(architecture_config).get_model()
    print_colored(f"> model {architecture_name} successfully loaded!\n", color="red")
    return