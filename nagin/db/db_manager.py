"""Main class for the management of the database ant datasets."""
import numpy as np
from qcodes import initialise_or_create_database_at
from qube.measurement.content import run_id_to_datafile

# from pathlib import Path


class DBManager():
    """Manager that opens and controls the database.

        Parameters
        ----------
        db_file: Path, Str.
            Path to the database file.

        run_id: Int.
            Id of the initial database run to retrieve.

        """

    def __init__(
        self, db_file=None,  run_id=None
    ):
        self.db_file = None
        if db_file is not None:
            self.open_db(db_file)

        self.runs = {}
        self.run = None
        if run_id is not None:
            self.update_run(run_id)

    def open_db(self, db_file):
        """Open the given database.

        Parameters
        ----------
        db_file: Path, Str.
            Path to the database file.

        """
        self.db_file = db_file
        initialise_or_create_database_at(self.db_file)

    def update_run(self, run_id):
        """Update the active run.

        Parameters
        ----------
        run_id: Int.
            Id of the initial database run to retrieve.

        """
        self.run = self.runs.get(run_id, None)
        if self.run is None:
            self.run = self.runs[run_id] = DatabaseRun(run_id)

        return self.run


class DatabaseRun():
    """Class that represents and controls a run in the database.

    Parameters
    ----------
    run_id: Int.
        Id of the initial database run to retrieve.

    """

    def __init__(self, run_id, **kwargs):
        self.run_id = run_id
        self.extract_data(run_id, **kwargs)

    def extract_data(self, run_id, **kwargs):
        """Get the data from the database.

        Parameters
        ----------
        run_id: Int.
            Id of the initial database run to retrieve.

        Other Parameters
        ----------------
        datasets: Array of Str.
            List of datasets to retrieve.

        """
        self.df = run_id_to_datafile(run_id)

        axes = self.df[0].axes

        self.data = {"axes": {ax.name: ax.value for ax in axes}}

        for dataset in kwargs.pop("datasets", self.df.ds_names):
            try:
                self.data[dataset] = self.df.get_dataset(dataset).value
                print(
                    f"Dataset fetched: {dataset}. "
                    f"(Shape: {self.data[dataset].shape})"
                )
            except KeyError:
                print(
                    f"'{dataset}' is not a dataset in the table."
                )

        print(f"Axes: {self.data['axes'].keys()}.")
        return self.df, self.data

    def has(self, dataset):
        """Check that the dataset exists in the data.

        Parameters
        ----------
        dataset: Str.
            Name of the dataset to search.

        """
        return dataset in self.data

    def check_datasets(self, datasets):
        """Check that the datasets are in the data dictionary.

        Parameters
        ----------
        datasets: Str.
            Names of the datasets to search.

        """
        for dataset in datasets:
            if not self.has(dataset):
                raise KeyError(f"Dataset '{dataset}' not in the data fetched.")

    def get_interval_roi(self, dataset, i_val=None, f_val=None):
        """Get a region of interest from an interval inclusive.

        Parameters
        ----------
        dataset: Str.
            Name of the dataset in which get the mask of the ROI.

        i_val: Float | Str.
            Initial value of the ROI. If String, it search the value in the
            data dict.

        f_val: Float | Str.
            Final value of the ROI. If String, it search the value in the data
            dict.

        """

        if isinstance(dataset, str):
            vals = self.data[dataset]
        else:
            vals = dataset

        if i_val is None:
            i_val = vals.min()
        elif isinstance(i_val, str):
            i_val = self.data[i_val]

        if f_val is None:
            f_val = vals.max()
        elif isinstance(f_val, str):
            f_val = self.data[f_val]

        return np.logical_and(i_val <= vals, f_val >= vals)

    def __getitem__(self, dataset):
        """Get the desired set of the data.

        Parameters
        ----------
        dataset: Str.
            Name of the dataset to retrieve.

        """
        return self.data[dataset]

    def __setitem__(self, dataset, value):
        """Set a value to the desired set of the data.

        Parameters
        ----------
        dataset: Str.
            Name of the dataset to modify.

        value: Any.
            Value to be set.

        """
        self.data[dataset] = value

    def ax(self, axis=None):
        """Get the desired axis.

        Parameters
        ----------
        axis: Str (Optional).
            Name of the axis to retrieve.

        """
        if axis is not None:
            return self.data["axes"][axis]

        return self.data["axes"]

    def scale_dataset(self, dataset, factor):
        """Scale a dataset.

        Parameters
        ----------
        dataset: Str.
            Name of the dataset to modify.

        factor: Any.
            Factor to increment the dataset.

        """
        return self.data[dataset] * factor
