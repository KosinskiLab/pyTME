import argparse
import tempfile
import subprocess
from shutil import rmtree
from os.path import exists
from os import remove, makedirs
from pathlib import Path

import pytest
import numpy as np

from tme import Density, Orientations
from tme.backends import backend as be
from tme.matching_utils import generate_tempfile_name

np.random.seed(42)
available_backends = tuple(x for x in be.available_backends() if x != "mlx")


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


class TestSetup:
    @classmethod
    def setup_class(cls):
        target = np.random.rand(20, 20, 20)
        template = np.random.rand(5, 5, 5)

        target_mask = 1.0 * (target > 0.5)
        template_mask = 1.0 * (template > 0.5)

        cls.target_path = generate_tempfile_name(suffix=".mrc")
        cls.template_path = generate_tempfile_name(suffix=".mrc")
        cls.target_mask_path = generate_tempfile_name(suffix=".mrc")
        cls.template_mask_path = generate_tempfile_name(suffix=".mrc")
        cls.tempdir = tempfile.mkdtemp()
        makedirs(cls.tempdir, exist_ok=True)

        orientations = Orientations(
            translations=((10, 10, 10), (12, 10, 15)),
            rotations=((0, 0, 0), (45, 12, 90)),
            metadata={
                "_pytmeScore": np.zeros(2, dtype=np.float32),
                "_rlnClassNumber": np.full(2, -1),
            },
        )
        cls.orientations_path = generate_tempfile_name(suffix=".star")
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
        target_path: str,
        template_path: str,
        template_mask_path: str,
        target_mask_path: str,
        use_target_mask: bool = False,
        backend: str = "numpyfftw",
        test_rejection_sampling: bool = False,
        background_correction: str = None,
    ):
        output_path = generate_tempfile_name(suffix=".pickle")

        argdict = {
            "-m": target_path,
            "-i": template_path,
            "-n": 1,
            "-a": 60,
            "-o": output_path,
            "--pad-edges": False,
            "--backend": backend,
        }

        if use_template_mask:
            argdict["--template-mask"] = template_mask_path

        if use_target_mask:
            argdict["--target-mask"] = target_mask_path

        if test_rejection_sampling:
            argdict["--orientations"] = self.orientations_path
            argdict["--orientations-uncertainty"] = 5
            argdict["--orientations-scaling"] = 1

        if background_correction is not None:
            argdict["--background-correction"] = background_correction

        if test_filter:
            argdict["--lowpass"] = 30
            argdict["--defocus"] = 3000
            argdict["--tilt-angles"] = "40,40"
            argdict["--wedge-axes"] = "2,0"
            argdict["--whiten"] = True

        cmd = argdict_to_command(argdict, executable="match_template")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        print(ret)
        assert ret.returncode == 0
        return output_path


class TestMatchTemplate(TestSetup):

    @pytest.mark.parametrize("backend", available_backends)
    @pytest.mark.parametrize("test_rejection_sampling", (False, True))
    @pytest.mark.parametrize("background_correction", (None, "phase-scrambling"))
    def test_match_template(
        self,
        backend: bool,
        test_rejection_sampling: bool,
        background_correction: str,
    ):
        self.run_matching(
            use_template_mask=True,
            use_target_mask=True,
            backend=backend,
            test_filter=True,
            template_path=self.template_path,
            target_path=self.target_path,
            template_mask_path=self.template_mask_path,
            target_mask_path=self.target_mask_path,
            test_rejection_sampling=test_rejection_sampling,
            background_correction=background_correction,
        )

    def test_match_template_symmetry(self):
        output_path = generate_tempfile_name(suffix=".pickle")
        argdict = {
            "-m": self.target_path,
            "-i": self.template_path,
            "-n": 1,
            "-a": 60,
            "--symmetry": "C4",
            "-o": output_path,
            "--pad-edges": False,
            "--backend": "numpyfftw",
        }
        cmd = argdict_to_command(argdict, executable="match_template")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        assert ret.returncode == 0, ret.stderr
        self.try_delete(output_path)

    def test_match_template_deprecated_cone_flag(self):
        output_path = generate_tempfile_name(suffix=".pickle")
        argdict = {
            "-m": self.target_path,
            "-i": self.template_path,
            "-n": 1,
            "-a": 60,
            "--cone-angle": 30,
            "-o": output_path,
            "--pad-edges": False,
            "--backend": "numpyfftw",
        }
        # Use the canonical `pytme match` entry so the only source of a deprecation
        # warning on stderr is the --cone-angle flag itself, not the legacy
        # `match_template` alias (which is always deprecated).
        cmd = argdict_to_command(argdict, executable="pytme match")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        assert ret.returncode == 0, ret.stderr
        assert b"--cone-angle is deprecated" in ret.stderr
        self.try_delete(output_path)

    def test_match_template_deprecated_memory_flag(self):
        output_path = generate_tempfile_name(suffix=".pickle")
        argdict = {
            "-m": self.target_path,
            "-i": self.template_path,
            "-n": 1,
            "-a": 60,
            "--memory": 1000000000,
            "-o": output_path,
            "--pad-edges": False,
            "--backend": "numpyfftw",
        }
        # `pytme match` isolates the deprecation warning to the --memory flag;
        # matching still succeeds using --memory-scaling.
        cmd = argdict_to_command(argdict, executable="pytme match")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        assert ret.returncode == 0, ret.stderr
        assert b"--memory is deprecated" in ret.stderr
        self.try_delete(output_path)


class TestPostprocessing(TestSetup):
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

        cls.score_pickle = cls.run_matching(cls, **matching_kwargs)

    @classmethod
    def teardown_class(cls):
        cls.try_delete(cls.score_pickle)

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
            "--input-file": self.score_pickle,
            "--output-format": "orientations",
            "--output-prefix": f"{self.tempdir}/temp",
            "--peak-oversampling": peak_oversampling,
            "--num-peaks": 3,
        }

        if score_cutoff is not None:
            if len(score_cutoff) == 1:
                argdict["--n-false-positives"] = 1
            else:
                min_score, max_score = score_cutoff
                argdict["--min-score"] = min_score
                argdict["--max-score"] = max_score

        match distance_cutoff_strategy:
            case 1:
                argdict["--mask-edges"] = True
            case 2:
                argdict["--min-distance"] = 5
            case 3:
                argdict["--min-boundary-distance"] = 5

        cmd = argdict_to_command(argdict, executable="postprocess")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        print(ret)
        assert ret.returncode == 0

    @pytest.mark.parametrize(
        "output_format",
        ("orientations", "alignment", "relion4", "relion5"),
    )
    def test_postproces_score_formats(self, output_format):
        self.try_delete(self.tempdir)
        makedirs(self.tempdir, exist_ok=True)

        argdict = {
            "--input-file": self.score_pickle,
            "--output-format": output_format,
            "--output-prefix": f"{self.tempdir}/temp",
            "--num-peaks": 3,
            "--peak-caller": "PeakCallerMaximumFilter",
        }
        cmd = argdict_to_command(argdict, executable="postprocess")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        print(ret)

        match output_format:
            case "orientations":
                assert exists(f"{self.tempdir}/temp.tsv")
            case "alignment":
                assert exists(f"{self.tempdir}/temp_0.mrc")
            case "relion4":
                assert exists(f"{self.tempdir}/temp.star")
                with open(f"{self.tempdir}/temp.star", encoding="utf-8") as f:
                    text = f.read()
                assert "data_optics" in text
                assert "_rlnImagePixelSize" in text
                assert "5.0" in text
            case "relion5":
                assert exists(f"{self.tempdir}/temp.star")
                with open(f"{self.tempdir}/temp.star", encoding="utf-8") as f:
                    text = f.read()
                assert "data_optics" in text
                assert "_rlnImagePixelSize" in text
                assert "5.0" in text
            case "pickle":
                assert exists(f"{self.tempdir}/temp.pickle")
        assert ret.returncode == 0

    def test_postprocess_score_local_optimization(self):
        self.try_delete(self.tempdir)
        makedirs(self.tempdir, exist_ok=True)

        argdict = {
            "--input-file": self.score_pickle,
            "--output-format": "orientations",
            "--output-prefix": f"{self.tempdir}/temp",
            "--num-peaks": 1,
            "--local-optimization": True,
        }
        cmd = argdict_to_command(argdict, executable="postprocess")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        print(ret)
        assert ret.returncode == 0

    def test_postprocess_no_stats(self):
        self.try_delete(self.tempdir)
        makedirs(self.tempdir, exist_ok=True)
        argdict = {
            "--input-file": self.score_pickle,
            "--output-format": "orientations",
            "--output-prefix": f"{self.tempdir}/temp",
            "--num-peaks": 3,
            "--no-stats": True,
        }
        cmd = argdict_to_command(argdict, executable="postprocess")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        print(ret)
        assert ret.returncode == 0
        assert exists(f"{self.tempdir}/temp.tsv")

    def test_postprocess_target_mask_shape_mismatch(self):
        self.try_delete(self.tempdir)
        makedirs(self.tempdir, exist_ok=True)

        bad_mask = np.ones((10, 10, 10), dtype=np.float32)
        bad_mask_path = generate_tempfile_name(suffix=".mrc")
        Density(bad_mask, sampling_rate=5).to_file(bad_mask_path)

        argdict = {
            "--input-file": self.score_pickle,
            "--output-format": "orientations",
            "--output-prefix": f"{self.tempdir}/temp",
            "--target-mask": bad_mask_path,
            "--num-peaks": 3,
        }
        cmd = argdict_to_command(argdict, executable="postprocess")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        self.try_delete(bad_mask_path)
        assert ret.returncode != 0
        assert (
            b"target mask" in ret.stderr.lower()
            or b"shape mismatch" in ret.stderr.lower()
        )


class TestEstimateMemoryUsage(TestSetup):
    @classmethod
    def setup_class(cls):
        super().setup_class()

    @pytest.mark.parametrize("ncores", (1, 4, 8))
    @pytest.mark.parametrize("pad_edges", (False, True))
    def test_estimation_cli(self, ncores, pad_edges):

        argdict = {
            "-m": self.target_path,
            "-i": self.template_path,
            "--ncores": ncores,
            "--pad-edges": pad_edges,
            "--score": "FLCSphericalMask",
        }

        cmd = argdict_to_command(argdict, executable="estimate_memory_usage")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        assert ret.returncode == 0


class TestPreprocess(TestSetup):
    @classmethod
    def setup_class(cls):
        super().setup_class()

    @pytest.mark.parametrize("backend", available_backends)
    @pytest.mark.parametrize("align_axis", (False, True))
    @pytest.mark.parametrize("invert_contrast", (False, True))
    def test_preprocess_cli(self, backend, align_axis, invert_contrast):

        argdict = {
            "-m": self.target_path,
            "--backend": backend,
            "--lowpass": 40,
            "--sampling-rate": 5,
            "-o": f"{self.tempdir}/out.mrc",
        }
        if align_axis:
            argdict["--align-axis"] = 2

        if invert_contrast:
            argdict["--invert-contrast"] = True

        cmd = argdict_to_command(argdict, executable="preprocess")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        assert ret.returncode == 0

    def test_preprocess_scramble_is_deterministic(self):
        import numpy as _np
        from tme import Density as _Density

        out_a = f"{self.tempdir}/scram_a.mrc"
        out_b = f"{self.tempdir}/scram_b.mrc"
        out_plain = f"{self.tempdir}/plain.mrc"

        base = {
            "-m": self.target_path,
            "--sampling-rate": 5,
            "--lowpass": 0,
        }
        for out, extra in (
            (out_a, {"--scramble-phases": True, "--scramble-seed": 7}),
            (out_b, {"--scramble-phases": True, "--scramble-seed": 7}),
            (out_plain, {}),
        ):
            argdict = {**base, "-o": out, **extra}
            cmd = argdict_to_command(argdict, executable="preprocess")
            ret = subprocess.run(cmd, capture_output=True, shell=True)
            assert ret.returncode == 0, ret.stderr

        a = _Density.from_file(out_a).data
        b = _Density.from_file(out_b).data
        plain = _Density.from_file(out_plain).data
        assert _np.allclose(a, b)
        assert not _np.allclose(a, plain)


class TestPytmeCLI:
    """Tests for the unified ``pytme`` CLI entry point."""

    def test_pytme_version(self):
        ret = subprocess.run(["pytme", "--version"], capture_output=True, text=True)
        assert ret.returncode == 0
        assert "pytme" in ret.stdout

    def test_pytme_help_lists_subcommands(self):
        ret = subprocess.run(["pytme", "--help"], capture_output=True, text=True)
        assert ret.returncode == 0

    def test_pytme_match_help(self):
        ret = subprocess.run(
            ["pytme", "match", "--help"], capture_output=True, text=True
        )
        assert ret.returncode == 0
        # match_template's argparse should produce help mentioning target/template
        assert "pytme match" in ret.stdout

    def test_pytme_no_subcommand_shows_help(self):
        ret = subprocess.run(["pytme"], capture_output=True, text=True)
        assert ret.returncode == 0
        assert "pytme" in ret.stdout

    def test_deprecated_alias_emits_warning(self):
        from tme.scripts.cli import match_template_deprecated

        with pytest.warns(FutureWarning, match="pytme match"):
            try:
                match_template_deprecated()
            except SystemExit:
                pass


class TestRunner(TestSetup):
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

        cls.score_pickle = cls.run_matching(cls, **matching_kwargs)

    @classmethod
    def teardown_class(cls):
        super().teardown_class(cls)
        cls.try_delete(cls.score_pickle)

    def test_runner_bash_submission(self):
        argdict = {
            "--tomograms": self.target_path,
            "-i": self.template_path,
            "-a": 60,
            "-n": 1,
            "--output-dir": self.tempdir,
            "--submit-command": "bash",
            "--force": True,
        }
        cmd = argdict_to_command(argdict, executable="pytme batch matching")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        assert ret.returncode == 0

    def test_runner_analysis_bash_submission(self):
        self.try_delete(self.tempdir)
        makedirs(self.tempdir, exist_ok=True)

        # Get the directory containing the score pickle for the input pattern
        pickle_dir = str(Path(self.score_pickle).parent)
        pickle_pattern = f"{pickle_dir}/*.pickle"

        argdict = {
            "--input-files": pickle_pattern,
            "--output-dir": self.tempdir,
            "--submit-command": "bash",
            "--force": True,
            "--output-format": "relion4",
            "--num-peaks": 3,
        }
        cmd = argdict_to_command(argdict, executable="pytme batch analysis")
        ret = subprocess.run(cmd, capture_output=True, shell=True)
        assert ret.returncode == 0


class TestCliHelpers:
    def test_check_symmetry_accepts_cyclic_and_dihedral(self):
        from tme.utils.cli import check_symmetry

        assert check_symmetry("c4") == "C4"
        assert check_symmetry("D2") == "D2"

    def test_check_symmetry_rejects_bare_integer(self):
        from tme.utils.cli import check_symmetry

        with pytest.raises(argparse.ArgumentTypeError):
            check_symmetry("4")

    def test_deprecated_action_emits_futurewarning(self):
        from tme.utils.cli import DeprecatedAction

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--old", action=DeprecatedAction, replacement="Use --new instead."
        )
        with pytest.warns(FutureWarning, match="Use --new instead."):
            args = parser.parse_args(["--old", "5"])
        assert args.old is None


class TestParseRotationLogic:
    def test_symmetry_with_no_rotation_search_keeps_identity(self):
        from types import SimpleNamespace
        from tme.scripts.match_template import parse_rotation_logic

        args = SimpleNamespace(
            particle_diameter=None, angular_sampling=180.0, symmetry="C4"
        )
        rotations = parse_rotation_logic(args, ndim=3)
        assert rotations.shape == (1, 3, 3)
        assert np.allclose(rotations[0], np.eye(3))


class TestResolveOrientationScaling:
    def test_explicit_scaling_passthrough(self):
        from tme.scripts.match_template import _resolve_orientation_scaling

        assert _resolve_orientation_scaling(3.0, 5.0, is_mesh=False) == 3.0
        assert _resolve_orientation_scaling(3.0, 5.0, is_mesh=True) == 3.0

    def test_point_default_is_identity(self):
        from tme.scripts.match_template import _resolve_orientation_scaling

        assert _resolve_orientation_scaling(None, 5.0, is_mesh=False) == 1.0

    def test_mesh_default_is_max_sampling_rate(self):
        import numpy as np
        from tme.scripts.match_template import _resolve_orientation_scaling

        assert (
            _resolve_orientation_scaling(None, np.array([4.0, 5.0, 6.0]), is_mesh=True)
            == 6.0
        )
