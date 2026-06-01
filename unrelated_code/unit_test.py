import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())

from core.architectures.unit_test import architecture_unit_test
from core.datasets.unit_test import dataset_unit_test



if __name__ == "__main__":
    architecture_unit_test()
    dataset_unit_test()
    print("Done!")