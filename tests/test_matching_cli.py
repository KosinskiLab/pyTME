import tempfile
import subprocess
from shutil import rmtree
from os.path import exists
from os import remove, makedirs

import pytest
import numpy as np
from tme import Density, Orientations
from tme.backends import backend as be

np.random.seed(42)
available_backends = (x for x in be.available_backends() if x != "mlx")


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
        cls.tempdir = tempfile.TemporaryDirectory().name
        makedirs(cls.tempdir, exist_ok=True)

        orientations = Orientations(
            translations=((10, 10, 10), (12, 10, 15)),
            rotations=((0, 0, 0), (45, 12, 90)),
            scores=(0, 0),
            details=(-1, -1),
        )
        cls.orientations_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".star"
        ).name
        orientations.to_file(cls.orientations_path)

        Density(target, sampling_rate=5).to_file(cls.target_path)
        Density(template, sampling_rate=5).to_file(cls.template_path)
        Density(target_mask, sampling_rate=5).to_file(cls.target_mask_path)
        Density(template_mask, sampling_rate=5).to_file(cls.template_mask_path)

    def teardown_class(cls):
        cls.try_delete(cls.target_path)
        cls.try_delete(cls.template_path)
        cls.try_delete(cls.target_mask_path)
        cls.try_delete(cls.template_mask_path)
        cls.try_delete(cls.orientations_path)
        cls.try_delete(cls.tempdir)

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

    def run_matching(
        self,
        use_template_mask: bool,
        test_filter: bool,
        call_peaks: bool,
        target_path: str,
        template_path: str,
        template_mask_path: str,
        target_mask_path: str,
        use_target_mask: bool = False,
        backend: str = "numpyfftw",
        test_rejection_sampling: bool = False,
    ):
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix="pickle").name

        argdict = {
            "-m": target_path,
            "-i": template_path,
            "-n": 1,
            "-a": 60,
            "-o": output_path,
            "--pad_edges": False,
            "--backend": backend,
        }

        if use_template_mask:
            argdict["--template_mask"] = template_mask_path

        if use_target_mask:
            argdict["--target_mask"] = target_mask_path

        if test_rejection_sampling:
            argdict["--orientations"] = self.orientations_path

        if test_filter:
            argdict["--lowpass"] = 30
            argdict["--defocus"] = 3000
            argdict["--tilt_angles"] = "40,40"
            argdict["--wedge_axes"] = "2,0"
            argdict["--whiten"] = True

        if call_peaks:
            argdict["-p"] = True

        cmd = argdict_to_command(argdict, executable="match_template.py")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        print(ret)
        assert ret.returncode == 0
        return output_path

    @pytest.mark.parametrize("backend", available_backends)
    @pytest.mark.parametrize("call_peaks", (False, True))
    @pytest.mark.parametrize("use_template_mask", (False, True))
    @pytest.mark.parametrize("test_filter", (False, True))
    @pytest.mark.parametrize("test_rejection_sampling", (False, True))
    def test_match_template(
        self,
        backend: bool,
        call_peaks: bool,
        use_template_mask: bool,
        test_filter: bool,
        test_rejection_sampling: bool,
    ):
        if backend == "jax" and (call_peaks or test_rejection_sampling):
            return None

        self.run_matching(
            use_template_mask=use_template_mask,
            use_target_mask=True,
            backend=backend,
            test_filter=test_filter,
            call_peaks=call_peaks,
            template_path=self.template_path,
            target_path=self.target_path,
            template_mask_path=self.template_mask_path,
            target_mask_path=self.target_mask_path,
            test_rejection_sampling=test_rejection_sampling,
        )


class TestPostprocessing(TestMatchTemplate):
    @classmethod
    def setup_class(cls):
        super().setup_class()

        matching_kwargs = {
            "use_template_mask": False,
            "use_target_mask": False,
            "test_filter": False,
            "template_path": cls.template_path,
            "target_path": cls.target_path,
            "template_mask_path": cls.template_mask_path,
            "target_mask_path": cls.target_mask_path,
            "test_rejection_sampling": False,
        }

        cls.score_pickle = cls.run_matching(
            cls,
            call_peaks=False,
            **matching_kwargs,
        )
        cls.peak_pickle = cls.run_matching(cls, call_peaks=True, **matching_kwargs)

    @classmethod
    def teardown_class(cls):
        cls.try_delete(cls.score_pickle)
        cls.try_delete(cls.peak_pickle)

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
            "--num_peaks": 3,
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
        ("orientations", "alignment", "average", "relion4", "relion5"),
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
            "--num_peaks": 3,
            "--peak_caller": "PeakCallerMaximumFilter",
        }
        cmd = argdict_to_command(argdict, executable="postprocess.py")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        print(ret)

        match output_format:
            case "orientations":
                assert exists(f"{self.tempdir}/temp.tsv")
            case "alignment":
                assert exists(f"{self.tempdir}/temp_0.mrc")
            case "average":
                assert exists(f"{self.tempdir}/temp.mrc")
            case "relion4":
                assert exists(f"{self.tempdir}/temp.star")
            case "relion5":
                assert exists(f"{self.tempdir}/temp.star")
            case "pickle":
                assert exists(f"{self.tempdir}/temp.pickle")
        assert ret.returncode == 0

    def test_postprocess_score_local_optimization(self):
        self.try_delete(self.tempdir)
        makedirs(self.tempdir, exist_ok=True)

        argdict = {
            "--input_file": self.score_pickle,
            "--output_format": "orientations",
            "--output_prefix": f"{self.tempdir}/temp",
            "--num_peaks": 1,
            "--local_optimization": True,
        }
        cmd = argdict_to_command(argdict, executable="postprocess.py")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        assert ret.returncode == 0


class TestEstimateMemoryUsage(TestMatchTemplate):
    @classmethod
    def setup_class(cls):
        super().setup_class()

    @pytest.mark.parametrize("ncores", (1, 4, 8))
    @pytest.mark.parametrize("pad_edges", (False, True))
    def test_estimation(self, ncores, pad_edges):

        argdict = {
            "-m": self.target_path,
            "-i": self.template_path,
            "--ncores": ncores,
            "--pad_edges": pad_edges,
            "--score": "FLCSphericalMask",
        }

        cmd = argdict_to_command(argdict, executable="estimate_memory_usage.py")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        assert ret.returncode == 0


class TestPreprocess(TestMatchTemplate):
    @classmethod
    def setup_class(cls):
        super().setup_class()

    @pytest.mark.parametrize("backend", available_backends)
    @pytest.mark.parametrize("align_axis", (False, True))
    @pytest.mark.parametrize("invert_contrast", (False, True))
    def test_estimation(self, backend, align_axis, invert_contrast):

        argdict = {
            "-m": self.target_path,
            "--backend": backend,
            "--lowpass": 40,
            "--sampling_rate": 5,
            "-o": f"{self.tempdir}/out.mrc",
        }
        if align_axis:
            argdict["--align_axis"] = 2

        if invert_contrast:
            argdict["--invert_contrast"] = True

        cmd = argdict_to_command(argdict, executable="preprocess.py")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        assert ret.returncode == 0
