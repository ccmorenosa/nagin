"""Main data analyzer class."""
# from pathlib import Path
import hashlib
import hmac
import pickle

import numpy as np
from pretty_verbose import Logger
from scipy.constants import e, h, k, physical_constants

from nagin.db import DBManager
from nagin.tools.stats import S_QPC_LC

ureg = pint.get_application_registry()
Q_ = ureg.Quantity


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

    G0 = physical_constants["conductance quantum"][0] * ureg("S")
    R0 = 1/G0
    e = e * ureg("C")
    h = h * ureg("J Hz^-1")
    k = k * ureg("J K^-1")

    S_defaults = {
        "T": 13e-3 * ureg("K"),
        "G": 483.6,
        "S_A": 4*k,
        "S_B": 8.3e-07 * ureg("V^2"),
        "S_C": 3.5e-14 * ureg("V^2 Hz^-1"),
        "alpha": 1,
        "Rqpc": R0,
        "L": 2.2e-6 * ureg("H"),
        "C": 182e-12 * ureg("F"),
        "Rp": 1.5 * ureg("ohm"),
    }

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

    def get_ureg(self):
        """Return the units registry for the application."""
        return pint.get_application_registry()

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

    def pinch_off_preconfig(self, dc=False, R_amp=1e6):
        """Configure data to work for shot noise."""
        self.run.check_datasets(
            ["a1_x", "a1_y", "a1_r"] + (["a1_dc"] if dc else [])
        )

        # Factor due to the lock-in (squared signal to sine).
        sqr_sin_factor = np.sqrt(2) / np.pi / R_amp
        self.run["x"] = self.run.scale_dataset("a1_x", sqr_sin_factor)
        self.run["y"] = self.run.scale_dataset("a1_y", sqr_sin_factor)
        self.run["r"] = self.run.scale_dataset("a1_r", sqr_sin_factor)

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

    def get_fitting_func(self, *default_vars):
        """Get a function to fit according the defaults variables."""
        S_fix_args = {}
        S_args = []

        for var, value in self.S_defaults.items():
            if var in default_vars:
                S_fix_args[var] = value
            else:
                S_args.append(var)

        def S_fit(f, Rqpc, *args):
            for i, arg in enumerate(args):
                S_fix_args[S_args[i]] = arg

            return S_QPC_LC(f, Rqpc=Rqpc, **S_fix_args)

        return S_fit

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
