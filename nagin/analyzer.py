"""Main data analyzer class."""
# from pathlib import Path
import hashlib
import hmac
import pickle
from itertools import product

import numpy as np
from pretty_verbose import Logger
from scipy.constants import e, h, k, physical_constants

from nagin.db import DBManager


class NaginAnalyzer(DBManager):
    """Data processing and analyzing class.

    Parameters
    ----------
    db_file: Path, Str.
        Path to the database file.

    run_id: Int.
        Id of the initial database run to retrieve.

    Other Parameters
    ----------------
    logger: Bool.
        Indicates whether use or not a logger for the analysis.

    """

    G0 = physical_constants["conductance quantum"][0]
    R0 = 1/G0
    e = e
    h = h
    k = k

    def __init__(self, working_dir, db_file, run_id=None, **kwargs):
        self.working_dir = working_dir
        super().__init__(working_dir/db_file, run_id)

        self.logger = None
        if kwargs.pop("logger", False):
            self.logger = Logger(
                kwargs.pop("verbose", 3), kwargs.pop("name", "Sauron_C"),
                log_dir=kwargs.pop("log_dir", "logger"), overwrite=False
            )

        self.__key = None
        if kwargs.get("key", None) is not None:
            self.load_key(kwargs.pop("key"))

    def load_key(self, key_file):
        """Load a key."""
        with open(key_file, "rb") as kf:
            self.__key = kf.read()

    def save_pickle(self, filename):
        """Save the desired data pickled in the file."""
        pickled_data = pickle.dumps(self.run.data)

        if self.__key is not None:
            signature = hmac.new(
                self.__key, pickled_data, hashlib.sha256
            ).digest()
            content = signature + pickled_data
        else:
            content = pickled_data

        with open(filename, "wb") as f:
            f.write(content)

    def load_pickle_data(self, pickled_file):
        """Read the pickled file to extract the data."""
        with open(pickled_file, "rb") as f:
            file_content = f.read()

        if self.__key is not None:
            sig_size = hashlib.sha256().digest_size
            signature = file_content[:sig_size]
            pickled_data = file_content[sig_size:]

            expected_sig = hmac.new(
                self.__key, pickled_data, hashlib.sha256
            ).digest()

            if not hmac.compare_digest(signature, expected_sig):
                raise ValueError(
                    "HMAC validation failed! File was tampered or wrong key!"
                )

        else:
            pickled_data = file_content

        return pickle.dumps(pickled_data)

    def pinch_off_preconfig(self, dc=False):
        """Configure data to work for shot noise."""
        self.run.check_datasets(
            ["a1_x", "a1_y", "a1_r"] + (["a1_dc"] if dc else [])
        )

        # Factor due to the lock-in (squared signal to sine).
        sqr_sin_factor = np.sqrt(2) / np.pi / 1e6
        self.run.scale_dataset("a1_x", sqr_sin_factor)
        self.run.scale_dataset("a1_y", sqr_sin_factor)
        self.run.scale_dataset("a1_r", sqr_sin_factor)

    def shot_noise_preconfig(self, fr_start=0, fr_end=None, temperature=False):
        """Configure data to work for shot noise."""
        self.run.check_datasets(
            ["frequencies", "PSD_ch0"] +
            (["actual_mxc_temperature"] if temperature else [])
        )

        self.run["fr_i"] = fr_start
        self.run["fr_f"] = (
            fr_end if fr_end is not None else self.run["frequencies"].max()
        )

        mask = self.run.get_interval_roi("frequencies", "fr_i", "fr_f")
        self.run["ROI_mask"] = mask

        self.run["fr_ROI"] = self.run["frequencies"][mask]
        self.run["PSD_ROI"] = self.run["PSD_ch0"][mask]

        if temperature:
            self.run["T_ROI"] = self.run["actual_mxc_temperature"][mask]

    def shot_noise_fits(self, axes=(), norm_ids=None):
        """Run fits on the shot noise data at the given axes."""
        fr = self.run["fr_ROI"]
        PSD = self.run["PSD_ROI"]

        axes_data = [
            enumerate(self.run["axes"][ax]) for ax in axes
        ]

        for iteration in product(*axes_data):
            pass

    def __getitem__(self, dataset):
        """Get the desired set of the data.

        Parameters
        ----------
        dataset: Str.
            Name of the dataset to retrieve.

        """
        return self.run[dataset]

    def __setitem__(self, dataset, value):
        """Set a value to the desired set of the data.

        Parameters
        ----------
        dataset: Str.
            Name of the dataset to modify.

        value: Any.
            Value to be set.

        """
        self.run[dataset] = value
