from itertools import product
from typing import Callable, Dict, Iterable

from tqdm.auto import tqdm


def sweep(fn: Callable, params: Dict[str, Iterable]) -> None:
    """
    Run a sweep of a function over various settings.

    Given a function and a dictionary of parameters, run the function over all
    combinations of the parameters.

    Parameters
    ----------
    fn : Callable
        Function to run.

    params : Dict[str, Iterable]
        Dictionary of parameters to run the function over. The keys are the
        parameter names and the values are the values to run the function with.
    """
    all_combinations = list(product(*params.values()))
    for combination in tqdm(all_combinations):
        fn(**dict(zip(params.keys(), combination)))
