from collections import ChainMap
from csv import DictReader, DictWriter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Tuple, Union
from warnings import warn

from promptquality.constants.dataset_format import DatasetFormat
from promptquality.utils.dependencies import is_pyarrow_available

DatasetType = Union[Path, str, Dict[str, List[str]], List[Dict[str, str]]]


def transpose_dict_to_rows(dataset: Dict[str, List[str]]) -> List[Dict[str, str]]:
    key_rows = [[{key: value} for value in values] for key, values in dataset.items()]
    return [dict(ChainMap(*key_row)) for key_row in zip(*key_rows)]


def parse_dataset(dataset: DatasetType) -> Tuple[Path, DatasetFormat]:
    """
    Set the dataset path and format.

    If the dataset is a dictionary or list, it will be stored as a a temporary file and
    the path to the temporary file will be returned. If the dataset is a path or string,
    the path will be returned. If the dataset is not a dictionary, list, path, or
    string, a ValueError will be raised.

    If the dataset is a dictionary or list and PyArrow is available, the dataset will
    be stored as a Feather file. Otherwise, the dataset will be stored as a CSV file.

    Parameters
    ----------
    dataset : DatasetType
        Dataset as provided by the user.

    Returns
    -------
    Tuple[Path, DatasetFormat]
        Path to the dataset file and the dataset format.

    Raises
    ------
    ValueError
        If the dataset is not a dictionary, list, path, or string.
    ValueError
        If the dataset file is not a CSV or Feather file.
    FileNotFoundError
        If the dataset file does not exist.
    """
    # Set the dataset path.
    dataset_path: Optional[Path] = None
    if isinstance(dataset, (dict, list)):
        dataset_rows = transpose_dict_to_rows(dataset) if isinstance(dataset, dict) else dataset
        if is_pyarrow_available:
            with NamedTemporaryFile(mode="wb", suffix=f".{DatasetFormat.feather}", delete=False) as dataset_file:
                from pyarrow import Table  # type: ignore[import]
                from pyarrow.feather import write_feather  # type: ignore[import]

                write_feather(Table.from_pylist(dataset_rows), dataset_file.name)
                dataset_path = Path(dataset_file.name)
        else:
            with NamedTemporaryFile(mode="wt", suffix=f".{DatasetFormat.csv}", delete=False) as dataset_file:
                field_names = dataset_rows[0].keys()
                writer = DictWriter(dataset_file, fieldnames=field_names)
                writer.writeheader()
                writer.writerows(dataset_rows)
                dataset_path = Path(dataset_file.name)
    elif isinstance(dataset, Path):
        dataset_path = dataset
    elif isinstance(dataset, str):
        dataset_path = Path(dataset)
    else:
        raise ValueError(f"Invalid dataset type: {type(dataset)}.")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file {dataset_path} does not exist.")

    # Set the dataset format.
    if dataset_path.suffix.endswith(DatasetFormat.csv):
        dataset_format = DatasetFormat.csv
        # Check if we have embedding columns.
        reader = DictReader(dataset_path.open("rt", errors="ignore"))
        for column in reader.fieldnames or []:
            if column.endswith("_embedding"):
                warn(
                    f"Column {column} appears to be an embedding column, please use "
                    "a feather file as dataset format instead."
                )
    elif dataset_path.suffix.endswith(DatasetFormat.jsonl):
        dataset_format = DatasetFormat.jsonl
    elif dataset_path.suffix.endswith(DatasetFormat.feather):
        dataset_format = DatasetFormat.feather
    else:
        raise ValueError(f"Dataset file {dataset_path} must be a CSV or Feather file.")

    return (dataset_path, dataset_format)
