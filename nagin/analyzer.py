"""Main data analyzer class."""
# from pathlib import Path
import hashlib
import hmac
import pickle

import numpy as np
import pint
from pretty_verbose import Logger
from scipy.constants import e, h, k, physical_constants
from scipy.interpolate import CubicSpline

from nagin.db import DBManager
from nagin.tools.stats import S_QPC_LC, Z_QPC_LC
from nagin.tools.data_manipulation import PSD_fit

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

        self.offset = kwargs.pop("offset", 0)
        self.threshold = kwargs.pop("threshold", 1.3e-14)
        self.repeat = kwargs.pop("repeat", 1)
        self.second_der = kwargs.pop("second_der", False)

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

    def compute_pinchoff_curve(
        self, v_qpc="V_qpc1", v_biases="ch19_square_amplitude",
        v_bias_type="array",
        **kwargs
    ):
        """Compute the pinch-off curve.

        Parameters
        ----------
        v_qpc: Str.
            Name of the axis with the V_qpc.

        v_biases: Str | Array | Integer.
            Value of the v_bias.

        v_bias_type: "index" | "value" | "array".
            Type of v_biases.

        """
        def compute_G(r_val, v_bias_val, r0):
            return r_val/(-v_bias_val*1e-3*(1-r_val/r0))

        # Getting data.
        r = kwargs.pop("r", self.run["r"])

        if isinstance(v_qpc, str):
            v_qpc = self.run["axes"][v_qpc]

        ROI = self.run.get_interval_roi(
            v_qpc, kwargs.pop("min_Vqpc", -1.99), kwargs.pop("max_Vqpc", -0.9)
        )

        if v_bias_type == "array":
            if isinstance(v_biases, str):
                v_biases = self.run["axes"][v_biases]

            pinch_off_Vqpc = np.empty((len(v_biases), len(v_qpc[ROI])))
            pinch_off_G = np.empty((len(v_biases), len(v_qpc[ROI])))

            for index, v_bias in enumerate(v_biases):
                pinch_off_Vqpc[index] = v_qpc[ROI]
                pinch_off_G[index] = compute_G(
                    r[ROI, index], v_bias, r[-1, index]
                )
                self.run["po_tao"] = CubicSpline(
                    pinch_off_Vqpc[index], pinch_off_G[index] / self.G0
                )
                self.run["po_R_f"] = CubicSpline(
                    pinch_off_Vqpc[index], 1 / pinch_off_G[index]
                )

        elif v_bias_type == "value":
            pinch_off_Vqpc = np.array(v_qpc[ROI])
            pinch_off_G = np.array(compute_G(r[ROI], v_biases, r[-1]))

            self.run["po_tao"] = CubicSpline(
                pinch_off_Vqpc, pinch_off_G / self.G0
            )
            self.run["po_R_f"] = CubicSpline(
                pinch_off_Vqpc, 1 / pinch_off_G
            )

        pinch_off_Vqpc = pinch_off_Vqpc * v_qpc[ROI].units
        pinch_off_G = pinch_off_G * ureg("S")

        self.run["po_Vqpc"] = pinch_off_Vqpc
        self.run["po_G"] = pinch_off_G
        self.run["po_G_G0"] = pinch_off_G / self.G0
        self.run["po_R"] = (1 / pinch_off_G).to("ohm")
        self.run["po_R_R0"] = self.R0 / pinch_off_G

    def shot_noise_preconfig(
        self, fr_start=0, fr_end=None, temperature=False, **config
    ):
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

        self.run["R"] = config.pop("R", lambda v: self.S_defaults["Rqpc"])

    def compute_noise_fit(self, fr, PSD, *fit_vars, **config):
        """Fit a noise curve to the data."""
        fix_params = {}
        for param in self.S_defaults.keys():
            if param in config:
                fix_params[param] = config.pop(param)

        res = self.get_fitting_func(*fit_vars, **fix_params)

        S_function, free_params, fix_values = res

        return PSD_fit(
            fr.copy(), PSD.copy(), config.pop("offset", self.offset),
            threshold=config.pop("threshold", self.threshold),
            repeat=config.pop("repeat", self.repeat),
            second_der=config.pop("second_der", self.second_der),
            func=S_function, **config
        ), free_params, fix_values

    def shotnoise_Vqpc_Vbias(
        self, *fit_vars, v_qpc="V_qpc1", v_biases="V_bias", **config
    ):
        """Compute shot noise for V _bias and V_qpc values and"""
        config.pop("full_output", None)
        fit_vars = list(fit_vars)
        if "R_func" not in fit_vars:
            R_func = config.pop("R_func", self.run["R"])
        else:
            # R = lambda v: self.S_defaults["R_func"]
            def R_func(_):
                return self.S_defaults["R_func"]
            fit_vars.remove("R_func")

        ROI_mask = config.pop("ROI_mask", self.run["ROI_mask"][:, 0, 0])
        fr = config.pop("fr", self.run["frequencies"][ROI_mask])
        PSD = config.pop("PSD", self.run["PSD_ch0"][ROI_mask])

        if isinstance(v_qpc, str):
            v_qpc = self.run["axes"][v_qpc]

        if isinstance(v_biases, str):
            v_biases = self.run["axes"][v_biases]

        shotnoise_res = np.empty((len(v_biases), len(v_qpc)), dtype=dict)
        for cut, v_qpc_val in enumerate(v_qpc):
            Rqpc = Q_(R_func(v_qpc_val)).m * ureg("ohm")

            for index, vbias in enumerate(v_biases):
                fr_data = fr[:, index, cut]
                PSD_data = PSD[:, index, cut]

                (mask, popt), free_params, fix_values = self.compute_noise_fit(
                    fr_data, PSD_data, *fit_vars, full_output=False, Rqpc=Rqpc,
                    **config
                )

                resonance = None
                shotnoise_fit = None
                maxima = None
                impedance = None
                impedance_resonante = None
                if popt is not None:
                    shotnoise_fit = S_QPC_LC(
                        Q_(fr_data).m, **fix_values
                    ) * ureg("V^2 / Hz")

                    resonance = ureg("Hz") / (
                        2*np.pi*np.sqrt(fix_values["L"] * fix_values["C"])
                    )

                    maxima = S_QPC_LC(
                        resonance.m, **fix_values
                    ) * ureg("V^2 / Hz")

                    impedance = np.real(Z_QPC_LC(
                        Q_(fr_data).m, fix_values["Rqpc"],
                        fix_values["L"], fix_values["C"], fix_values["Rp"]
                    )) * ureg("ohm")

                    impedance_resonante = np.real(Z_QPC_LC(
                        resonance.m, fix_values["Rqpc"],
                        fix_values["L"], fix_values["C"], fix_values["Rp"]
                    )) * ureg("ohm")

                shotnoise_res[index, cut] = {
                    "mask": mask, "vbias": vbias, "fix_values": fix_values,
                    "free_params": free_params, "resonance": resonance,
                    "impedance": impedance,
                    "impedance_resonante": impedance_resonante,
                    "popt": popt, "fit": shotnoise_fit, "maxima": maxima,
                }

        self.run["shotnoise"] = shotnoise_res

    def shotnoise_temperature(
        self, *fit_vars, temperature="actual_mxc_temperature", **config
    ):
        """Compute shot noise for V _bias and V_qpc values and"""
        config.pop("full_output", None)
        config.pop("T", None)
        fit_vars = list(fit_vars)

        if "T" in fit_vars:
            fit_vars.remove("T")

        if "V_qpc" in config:
            V_qpc = config.pop("V_qpc")
            R_func = config.pop("R_func", self.run["R"])

            config["Rqpc"] = R_func(V_qpc)

        ROI_mask = config.pop("ROI_mask", self.run["ROI_mask"][:, 0])
        fr = config.pop("fr", self.run["frequencies"][ROI_mask])
        PSD = config.pop("PSD", self.run["PSD_ch0"][ROI_mask])

        if isinstance(temperature, str):
            temperature = self.run[temperature]

        shotnoise_res = np.empty((len(temperature)), dtype=dict)
        for index, T in enumerate(temperature):

            fr_data = fr[:, index]
            PSD_data = PSD[:, index]

            (mask, popt), free_params, fix_values = self.compute_noise_fit(
                fr_data, PSD_data, *fit_vars, full_output=False, T=T,
                **config
            )

            resonance = None
            shotnoise_fit = None
            maxima = None
            impedance = None
            impedance_resonante = None
            if popt is not None:
                shotnoise_fit = S_QPC_LC(
                    Q_(fr_data).m, **fix_values
                ) * ureg("V^2 / Hz")

                resonance = ureg("Hz") / (
                    2*np.pi*np.sqrt(fix_values["L"] * fix_values["C"])
                )

                maxima = S_QPC_LC(
                    resonance.m, **fix_values
                ) * ureg("V^2 / Hz")

                impedance = np.real(Z_QPC_LC(
                    Q_(fr_data).m, fix_values["Rqpc"],
                    fix_values["L"], fix_values["C"], fix_values["Rp"]
                )) * ureg("ohm")

                impedance_resonante = np.real(Z_QPC_LC(
                    resonance.m, fix_values["Rqpc"],
                    fix_values["L"], fix_values["C"], fix_values["Rp"]
                )) * ureg("ohm")

            shotnoise_res[index] = {
                "mask": mask, "T": T, "fix_values": fix_values,
                "free_params": free_params, "resonance": resonance,
                "impedance": impedance,
                "impedance_resonante": impedance_resonante,
                "popt": popt, "fit": shotnoise_fit, "maxima": maxima,
            }

        self.run["shotnoise"] = shotnoise_res

    def get_fitting_func(self, *fit_vars, **fix_params):
        """Get a function to fit according the defaults variables."""
        S_fix_args = {}
        S_args = []

        for var, value in self.S_defaults.items():
            if var in fit_vars:
                S_args.append(var)
            elif var in fix_params:
                S_fix_args[var] = Q_(fix_params[var]).m
            else:
                S_fix_args[var] = Q_(value).m

        def S_fit(f, *args):
            for i, arg in enumerate(args):
                S_fix_args[S_args[i]] = arg

            return S_QPC_LC(Q_(f).m, **S_fix_args)

        return S_fit, S_args, S_fix_args

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
