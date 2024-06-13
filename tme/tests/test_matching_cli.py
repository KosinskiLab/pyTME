import tempfile
import subprocess
from shutil import rmtree
from os.path import exists
from os import remove, makedirs

import pytest
import numpy as np
from tme import Density

BACKEND_CLASSES = ["NumpyFFTWBackend", "PytorchBackend", "CupyBackend", "MLXBackend"]
BACKENDS_TO_TEST = []

test_gpu = (False,)
for backend_class in BACKEND_CLASSES:
    try:
        BackendClass = getattr(
            __import__("tme.backends", fromlist=[backend_class]), backend_class
        )
        BACKENDS_TO_TEST.append(BackendClass(device="cpu"))
        if backend_class == "CupyBackend":
            if BACKENDS_TO_TEST[-1].device_count() >= 1:
                test_gpu = (False, True)
    except ImportError:
        print(f"Couldn't import {backend_class}. Skipping...")


def argdict_to_command(input_args, executable: str):
    ret = []
    for key, value in input_args.items():
        if value is None:
            continue
        elif isinstance(value, bool):
            if value:
                ret.append(key)
        else:
            ret.extend([key, value])

    ret = [str(x) for x in ret]
    ret.insert(0, executable)
    return " ".join(ret)


class TestMatchTemplate:
    @classmethod
    def setup_class(cls):
        np.random.seed(42)
        target = np.random.rand(20, 20, 20)
        template = np.random.rand(5, 5, 5)

        target_mask = 1.0 * (target > 0.5)
        template_mask = 1.0 * (template > 0.5)

        cls.target_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mrc").name
        cls.template_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mrc"
        ).name
        cls.target_mask_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mrc"
        ).name
        cls.template_mask_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mrc"
        ).name

        Density(target, sampling_rate=5).to_file(cls.target_path)
        Density(template, sampling_rate=5).to_file(cls.template_path)
        Density(target_mask, sampling_rate=5).to_file(cls.target_mask_path)
        Density(template_mask, sampling_rate=5).to_file(cls.template_mask_path)

    def teardown_class(cls):
        cls.try_delete(cls.target_path)
        cls.try_delete(cls.template_path)
        cls.try_delete(cls.target_mask_path)
        cls.try_delete(cls.template_mask_path)

    @staticmethod
    def try_delete(file_path: str):
        try:
            remove(file_path)
        except Exception:
            pass
        try:
            rmtree(file_path, ignore_errors=True)
        except Exception:
            pass

    @staticmethod
    def run_matching(
        use_template_mask: bool,
        use_gpu: bool,
        test_filter: bool,
        call_peaks: bool,
        target_path: str,
        template_path: str,
        template_mask_path: str,
        target_mask_path: str,
        use_target_mask: bool = False,
    ):
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix="pickle").name

        argdict = {
            "-m": target_path,
            "-i": template_path,
            "-n": 1,
            "-a": 60,
            "-o": output_path,
            "--no_edge_padding": True,
            "--no_fourier_padding": True,
        }

        if use_template_mask:
            argdict["--template_mask"] = template_mask_path

        if use_target_mask:
            argdict["--target_mask"] = target_mask_path

        if use_gpu:
            argdict["--use_gpu"] = True

        if test_filter:
            argdict["--lowpass"] = 30
            argdict["--defocus"] = 3000
            argdict["--tilt_angles"] = "40,40:10"
            argdict["--wedge_axes"] = "0,2"
            argdict["--whiten"] = True

        if call_peaks:
            argdict["-p"] = True

        cmd = argdict_to_command(argdict, executable="match_template.py")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        assert ret.returncode == 0

        return output_path

    @pytest.mark.parametrize("call_peaks", (False, True))
    @pytest.mark.parametrize("use_template_mask", (False, True))
    @pytest.mark.parametrize("test_filter", (False, True))
    @pytest.mark.parametrize("use_gpu", test_gpu)
    def test_match_template(
        self,
        use_template_mask: bool,
        use_gpu: bool,
        test_filter: bool,
        call_peaks: bool,
    ):
        self.run_matching(
            use_template_mask=use_template_mask,
            use_target_mask=True,
            use_gpu=use_gpu,
            test_filter=test_filter,
            call_peaks=call_peaks,
            template_path=self.template_path,
            target_path=self.target_path,
            template_mask_path=self.template_mask_path,
            target_mask_path=self.target_mask_path,
        )


class TestPostprocessing(TestMatchTemplate):
    @classmethod
    def setup_class(cls):
        super().setup_class()

        matching_kwargs = {
            "use_template_mask": False,
            "use_target_mask": False,
            "use_gpu": False,
            "test_filter": False,
            "template_path": cls.template_path,
            "target_path": cls.target_path,
            "template_mask_path": cls.template_mask_path,
            "target_mask_path": cls.target_mask_path,
        }

        cls.score_pickle = cls.run_matching(
            call_peaks=False,
            **matching_kwargs,
        )
        cls.peak_pickle = cls.run_matching(call_peaks=True, **matching_kwargs)
        cls.tempdir = tempfile.TemporaryDirectory().name

    @classmethod
    def teardown_class(cls):
        cls.try_delete(cls.score_pickle)
        cls.try_delete(cls.peak_pickle)
        cls.try_delete(cls.tempdir)

    @pytest.mark.parametrize("distance_cutoff_strategy", (0, 1, 2, 3))
    @pytest.mark.parametrize("score_cutoff", (None, (1,), (0, 1), (None, 1), (0, None)))
    @pytest.mark.parametrize("peak_oversampling", (False, 4))
    def test_postprocess_score_orientations(
        self,
        peak_oversampling,
        score_cutoff,
        distance_cutoff_strategy,
    ):
        self.try_delete(self.tempdir)
        makedirs(self.tempdir, exist_ok=True)

        argdict = {
            "--input_file": self.score_pickle,
            "--output_format": "orientations",
            "--output_prefix": f"{self.tempdir}/temp",
            "--peak_oversampling": peak_oversampling,
            "--number_of_peaks": 3,
        }

        if score_cutoff is not None:
            if len(score_cutoff) == 1:
                argdict["--n_false_positives"] = 1
            else:
                min_score, max_score = score_cutoff
                argdict["--minimum_score"] = min_score
                argdict["--maximum_score"] = max_score

        match distance_cutoff_strategy:
            case 1:
                argdict["--mask_edges"] = True
            case 2:
                argdict["--min_distance"] = 5
            case 3:
                argdict["--min_boundary_distance"] = 5

        cmd = argdict_to_command(argdict, executable="postprocess.py")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        assert ret.returncode == 0

    @pytest.mark.parametrize("input_format", ("score", "peaks"))
    @pytest.mark.parametrize(
        "output_format",
        ("orientations", "alignment", "backmapping", "average", "relion"),
    )
    def test_postproces_score_formats(self, input_format, output_format):
        self.try_delete(self.tempdir)
        makedirs(self.tempdir, exist_ok=True)

        input_file = self.score_pickle
        if input_format == "peaks":
            input_file = self.peak_pickle

        argdict = {
            "--input_file": input_file,
            "--output_format": output_format,
            "--output_prefix": f"{self.tempdir}/temp",
            "--number_of_peaks": 3,
        }
        cmd = argdict_to_command(argdict, executable="postprocess.py")
        ret = subprocess.run(cmd, capture_output=True, shell=True)

        match output_format:
            case "orientations":
                assert exists(f"{self.tempdir}/temp.tsv")
            case "alignment":
                assert exists(f"{self.tempdir}/temp_0.mrc")
            case "backmapping":
                assert exists(f"{self.tempdir}/temp_backmapped.mrc")
            case "average":
                assert exists(f"{self.tempdir}/temp_average.mrc")
            case "relion":
                assert exists(f"{self.tempdir}/temp.star")

        assert ret.returncode == 0

    def test_postprocess_score_local_optimization(self):
        self.try_delete(self.tempdir)
        makedirs(self.tempdir, exist_ok=True)

        argdict = {
            "--input_file": self.score_pickle,
            "--output_format": "orientations",
            "--output_prefix": f"{self.tempdir}/temp",
            "--number_of_peaks": 1,
            "--local_optimization": True,
        }
        cmd = argdict_to_command(argdict, executable="postprocess.py")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        assert ret.returncode == 0
