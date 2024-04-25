import tempfile
from os import remove
import subprocess

import pytest
import numpy as np
from tme import Density

BACKEND_CLASSES = ["NumpyFFTWBackend", "PytorchBackend", "CupyBackend", "MLXBackend"]
BACKENDS_TO_TEST = []

test_gpu = (False, )
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


class TestMatchTemplate:

    def setup_method(self):
        np.random.seed(42)
        target = np.random.rand(20,20,20)
        template = np.random.rand(5,5,5)

        target_mask = 1.0 * (target > .5)
        template_mask = 1.0 * (template > .5)

        self.target_path = tempfile.NamedTemporaryFile(delete = False, suffix = ".mrc").name
        self.template_path = tempfile.NamedTemporaryFile(delete = False,  suffix = ".mrc").name
        self.target_mask_path = tempfile.NamedTemporaryFile(delete = False, suffix = ".mrc").name
        self.template_mask_path = tempfile.NamedTemporaryFile(delete = False, suffix = ".mrc").name

        Density(target, sampling_rate = 5).to_file(self.target_path)
        Density(template, sampling_rate = 5).to_file(self.template_path)
        Density(target_mask, sampling_rate = 5).to_file(self.target_mask_path)
        Density(template_mask, sampling_rate = 5).to_file(self.template_mask_path)

    def teardown_method(self):
        remove(self.target_path)
        remove(self.template_path)
        remove(self.target_mask_path)
        remove(self.template_mask_path)

    @pytest.mark.parametrize("use_template_mask", (False, True))
    @pytest.mark.parametrize("use_target_mask", (False, True))
    @pytest.mark.parametrize("edge_padding", (False, True))
    @pytest.mark.parametrize("fourier_padding", (False, True))
    @pytest.mark.parametrize("use_gpu", test_gpu)
    def test_match_template(self,use_template_mask, use_target_mask, edge_padding,fourier_padding, use_gpu):
        output_path = tempfile.NamedTemporaryFile(delete = False, suffix = "mrc").name
        cmd = [
            "match_template.py",
            "-m", self.target_path,
            "-i", self.template_path,
            "-n", "1",
            "-a", "60",
            "-o", output_path,
        ]

        if use_template_mask:
            cmd.extend(["--template_mask", self.template_mask_path])

        if use_target_mask:
            cmd.extend(["--target_mask", self.target_mask_path])

        if not edge_padding:
            cmd.append("--no_edge_padding")

        if not fourier_padding:
            cmd.append("--no_fourier_padding")

        if use_gpu:
            cmd.append("--use_gpu")

        ret = subprocess.run(cmd, capture_output = True)
        assert ret.returncode == 0
