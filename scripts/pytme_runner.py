#!python3
"""
PyTME Batch Runner - Refactored Core Classes
"""
import re
import argparse
import subprocess
from abc import ABC, abstractmethod

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from tme.backends import backend as be
from tme.cli import print_entry, print_block, sanitize_name


@dataclass
class TomoFiles:
    """Container for all files related to a single tomogram."""

    #: Tomogram identifier.
    tomo_id: str
    #: Path to tomogram.
    tomogram: Path
    #: XML file with tilt angles, defocus, etc.
    metadata: Path
    #: Path to tomogram mask, optional.
    mask: Optional[Path] = None

    def __post_init__(self):
        """Validate that required files exist."""
        if not self.tomogram.exists():
            raise FileNotFoundError(f"Tomogram not found: {self.tomogram}")
        if not self.metadata.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata}")
        if self.mask and not self.mask.exists():
            raise FileNotFoundError(f"Mask not found: {self.mask}")


class TomoDatasetDiscovery:
    """Find and match tomogram files using glob patterns."""

    def __init__(
        self,
        mrc_pattern: str,
        metadata_pattern: str,
        mask_pattern: Optional[str] = None,
    ):
        """
        Initialize with glob patterns for file discovery.

        Parameters
        ----------
        mrc_pattern: str
            Glob pattern for tomogram files, e.g., "/data/tomograms/*.mrc"
        metadata_pattern: str
            Glob pattern for metadata files, e.g., "/data/metadata/*.xml"
        mask_pattern: str
            Optional glob pattern for mask files, e.g., "/data/masks/*.mrc"
        """
        self.mrc_pattern = mrc_pattern
        self.metadata_pattern = metadata_pattern
        self.mask_pattern = mask_pattern

    @staticmethod
    def parse_id_from_filename(filename: str) -> str:
        """Extract the tomogram ID from filename by removing technical suffixes."""
        base = Path(filename).stem
        # Remove technical suffixes (pixel size, binning, filtering info)
        # Examples: "_10.00Apx", "_4.00Apx", "_bin4", "_dose_filt"
        base = re.sub(r"_\d+(\.\d+)?(Apx|bin\d*|dose_filt)$", "", base)

        # Remove common organizational prefixes if they exist
        for prefix in ["rec_Position_", "Position_", "rec_", "tomo_"]:
            if base.startswith(prefix):
                base = base[len(prefix) :]
                break
        return base

    def _create_mapping_table(self, pattern: str) -> Dict:
        """Create a mapping table between tomogram ids and file paths."""
        if pattern is None:
            return {}

        ret = {}
        path = Path(pattern).absolute()
        for file in list(Path(path.parent).glob(path.name)):
            file_id = self.parse_id_from_filename(file.name)
            if file_id not in ret:
                ret[file_id] = []
            ret[file_id].append(file)

        # This could all be done in one line but we want the messages.
        for key in ret.keys():
            value = ret[key]
            if len(value) > 1:
                print(f"Found id {key} multiple times at {value}. Using {value[0]}.")
            ret[key] = value[0]
        return ret

    def discover_tomograms(
        self, tomo_list: Optional[List[str]] = None, require_mask: bool = False
    ) -> List[TomoFiles]:
        """Find all matching tomogram files."""
        mrc_files = self._create_mapping_table(self.mrc_pattern)
        meta_files = self._create_mapping_table(self.metadata_pattern)
        mask_files = self._create_mapping_table(self.mask_pattern)

        if tomo_list:
            mrc_files = {k: v for k, v in mrc_files.items() if k in tomo_list}
            meta_files = {k: v for k, v in meta_files.items() if k in tomo_list}
            mask_files = {k: v for k, v in mask_files.items() if k in tomo_list}

        tomo_files = []
        for key, value in mrc_files.items():
            if key not in meta_files:
                print(f"No metadata for {key}, skipping it for now.")
                continue

            tomo_files.append(
                TomoFiles(
                    tomo_id=key,
                    tomogram=value,
                    metadata=meta_files[key],
                    mask=mask_files.get(key),
                )
            )
        return tomo_files


@dataclass
class TMParameters:
    """Template matching parameters."""

    template: Path
    template_mask: Optional[Path] = None

    # Angular sampling (auto-calculated or explicit)
    angular_sampling: Optional[float] = None
    particle_diameter: Optional[float] = None
    cone_angle: Optional[float] = None
    cone_sampling: Optional[float] = None
    axis_angle: float = 360.0
    axis_sampling: Optional[float] = None
    axis_symmetry: int = 1
    cone_axis: int = 2
    invert_cone: bool = False
    no_use_optimized_set: bool = False

    # Microscope parameters
    acceleration_voltage: float = 300.0  # kV
    spherical_aberration: float = 2.7e7  # Å
    amplitude_contrast: float = 0.07
    defocus: Optional[float] = None  # Å
    phase_shift: float = 0.0  # Dg

    # Processing options
    lowpass: Optional[float] = None  # Å
    highpass: Optional[float] = None  # Å
    pass_format: str = "sampling_rate"  # "sampling_rate", "voxel", "frequency"
    no_pass_smooth: bool = True
    interpolation_order: int = 3
    score_threshold: float = 0.0
    score: str = "FLCSphericalMask"

    # Weighting and correction
    tilt_weighting: Optional[str] = None  # "angle", "relion", "grigorieff"
    wedge_axes: str = "2,0"
    whiten_spectrum: bool = False
    scramble_phases: bool = False
    invert_target_contrast: bool = False

    # CTF parameters
    ctf_file: Optional[Path] = None
    no_flip_phase: bool = True
    correct_defocus_gradient: bool = False

    # Performance options
    centering: bool = False
    pad_edges: bool = False
    pad_filter: bool = False
    use_mixed_precision: bool = False
    use_memmap: bool = False

    # Analysis options
    peak_calling: bool = False
    num_peaks: int = 1000

    # Backend selection
    backend: str = "numpy"
    gpu_indices: Optional[str] = None

    # Reconstruction
    reconstruction_filter: str = "ramp"
    reconstruction_interpolation_order: int = 1
    no_filter_target: bool = False

    def __post_init__(self):
        """Validate parameters and convert units."""
        if not self.template.exists():
            raise FileNotFoundError(f"Template not found: {self.template}")
        if self.template_mask and not self.template_mask.exists():
            raise FileNotFoundError(f"Template mask not found: {self.template_mask}")
        if self.ctf_file and not self.ctf_file.exists():
            raise FileNotFoundError(f"CTF file not found: {self.ctf_file}")

        if self.tilt_weighting and self.tilt_weighting not in [
            "angle",
            "relion",
            "grigorieff",
        ]:
            raise ValueError(f"Invalid tilt weighting: {self.tilt_weighting}")

        if self.pass_format not in ["sampling_rate", "voxel", "frequency"]:
            raise ValueError(f"Invalid pass format: {self.pass_format}")

        valid_backends = list(be._BACKEND_REGISTRY.keys())
        if self.backend not in valid_backends:
            raise ValueError(
                f"Invalid backend: {self.backend}. Choose from {valid_backends}"
            )

    def to_command_args(
        self, tomo_files: TomoFiles, output_path: Path
    ) -> Dict[str, Any]:
        """Convert parameters to pyTME command arguments."""
        args = {
            "target": str(tomo_files.tomogram),
            "template": str(self.template),
            "output": str(output_path),
            "acceleration-voltage": self.acceleration_voltage,
            "spherical-aberration": self.spherical_aberration,
            "amplitude-contrast": self.amplitude_contrast,
            "interpolation-order": self.interpolation_order,
            "wedge-axes": self.wedge_axes,
            "score-threshold": self.score_threshold,
            "score": self.score,
            "pass-format": self.pass_format,
            "reconstruction-filter": self.reconstruction_filter,
            "reconstruction-interpolation-order": self.reconstruction_interpolation_order,
        }

        # Optional file arguments
        if self.template_mask:
            args["template-mask"] = str(self.template_mask)
        if tomo_files.mask:
            args["target-mask"] = str(tomo_files.mask)
        if tomo_files.metadata:
            args["ctf-file"] = str(tomo_files.metadata)
            args["tilt-angles"] = str(tomo_files.metadata)

        # Optional parameters
        if self.lowpass:
            args["lowpass"] = self.lowpass
        if self.highpass:
            args["highpass"] = self.highpass
        if self.tilt_weighting:
            args["tilt-weighting"] = self.tilt_weighting
        if self.defocus:
            args["defocus"] = self.defocus
        if self.phase_shift != 0:
            args["phase-shift"] = self.phase_shift
        if self.gpu_indices:
            args["gpu-indices"] = self.gpu_indices
        if self.backend != "numpy":
            args["backend"] = self.backend

        # Angular sampling
        if self.angular_sampling:
            args["angular-sampling"] = self.angular_sampling
        elif self.particle_diameter:
            args["particle-diameter"] = self.particle_diameter
        elif self.cone_angle:
            args["cone-angle"] = self.cone_angle
            if self.cone_sampling:
                args["cone-sampling"] = self.cone_sampling
            if self.axis_sampling:
                args["axis-sampling"] = self.axis_sampling
            if self.axis_angle != 360.0:
                args["axis-angle"] = self.axis_angle
            if self.axis_symmetry != 1:
                args["axis-symmetry"] = self.axis_symmetry
            if self.cone_axis != 2:
                args["cone-axis"] = self.cone_axis
        else:
            # Default fallback
            args["angular-sampling"] = 15.0

        args["num-peaks"] = self.num_peaks
        return args

    def get_flags(self) -> List[str]:
        """Get boolean flags for pyTME command."""
        flags = []
        if self.whiten_spectrum:
            flags.append("whiten-spectrum")
        if self.scramble_phases:
            flags.append("scramble-phases")
        if self.invert_target_contrast:
            flags.append("invert-target-contrast")
        if self.centering:
            flags.append("centering")
        if self.pad_edges:
            flags.append("pad-edges")
        if self.pad_filter:
            flags.append("pad-filter")
        if not self.no_pass_smooth:
            flags.append("no-pass-smooth")
        if self.use_mixed_precision:
            flags.append("use-mixed-precision")
        if self.use_memmap:
            flags.append("use-memmap")
        if self.peak_calling:
            flags.append("peak-calling")
        if not self.no_flip_phase:
            flags.append("no-flip-phase")
        if self.correct_defocus_gradient:
            flags.append("correct-defocus-gradient")
        if self.invert_cone:
            flags.append("invert-cone")
        if self.no_use_optimized_set:
            flags.append("no-use-optimized-set")
        if self.no_filter_target:
            flags.append("no-filter-target")
        return flags


@dataclass
class ComputeResources:
    """Compute resource requirements for a job."""

    cpus: int = 4
    memory_gb: int = 128
    gpu_count: int = 0
    gpu_type: Optional[str] = None  # e.g., "3090", "A100"
    time_limit: str = "05:00:00"
    partition: str = "gpu-el8"
    constraint: Optional[str] = None
    qos: str = "normal"

    def to_slurm_args(self) -> Dict[str, str]:
        """Convert to SLURM sbatch arguments."""
        args = {
            "ntasks": "1",
            "nodes": "1",
            "ntasks-per-node": "1",
            "cpus-per-task": str(self.cpus),
            "mem": f"{self.memory_gb}G",
            "time": self.time_limit,
            "partition": self.partition,
            "qos": self.qos,
            "export": "none",
        }

        if self.gpu_count > 0:
            args["gres"] = f"gpu:{self.gpu_count}"
            if self.gpu_type:
                args["constraint"] = f"gpu={self.gpu_type}"

        if self.constraint and not self.gpu_type:
            args["constraint"] = self.constraint

        return args


@dataclass
class TemplateMatchingTask:
    """A complete template matching task."""

    tomo_files: TomoFiles
    parameters: TMParameters
    resources: ComputeResources
    output_dir: Path

    @property
    def tomo_id(self) -> str:
        return self.tomo_files.tomo_id

    @property
    def output_file(self) -> Path:
        return self.output_dir / f"{self.tomo_id}.pickle"

    def create_output_dir(self) -> None:
        """Ensure output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


class ExecutionBackend(ABC):
    """Abstract base class for execution backends."""

    @abstractmethod
    def submit_job(self, task) -> str:
        """Submit a single job and return job ID or status."""
        pass

    @abstractmethod
    def submit_jobs(self, tasks: List) -> List[str]:
        """Submit multiple jobs and return list of job IDs."""
        pass


class SlurmBackend(ExecutionBackend):
    """SLURM execution backend for cluster job submission."""

    def __init__(
        self,
        dry_run: bool = False,
        script_dir: Path = Path("./slurm_scripts"),
        environment_setup: str = "module load pyTME",
    ):
        """
        Initialize SLURM backend.

        Parameters
        ----------
        dry_run : bool, optional
            Generate scripts but do not submit, defaults to False.
        script_dir: str, optional
            Directory to save generated scripts, defaults to ./slurm_scripts,
        environment_setup : str, optional
            Command to set up pyTME environment, defaults to module load pyTME.
        """
        self.dry_run = dry_run
        self.environment_setup = environment_setup
        self.script_dir = Path(script_dir) if script_dir else Path("./slurm_scripts")
        self.script_dir.mkdir(exist_ok=True, parents=True)

    def create_sbatch_script(self, task) -> Path:
        """Generate SLURM sbatch script for a template matching task."""
        script_path = self.script_dir / f"pytme_{task.tomo_id}.sh"

        # Ensure output directory exists
        task.create_output_dir()

        slurm_args = task.resources.to_slurm_args()
        slurm_args.update(
            {
                "output": f"{task.output_dir}/{task.tomo_id}_%j.out",
                "error": f"{task.output_dir}/{task.tomo_id}_%j.err",
                "job-name": f"pytme_{task.tomo_id}",
                "chdir": str(task.output_dir),
            }
        )

        script_lines = ["#!/bin/bash", "", "# SLURM directives"]
        for param, value in slurm_args.items():
            script_lines.append(f"#SBATCH --{param}={value}")

        script_lines.extend(
            [
                "",
                "# Environment setup",
                "\n".join(self.environment_setup.split(";")),
                "",
                "# Run template matching",
            ]
        )

        command_parts = ["match_template"]
        cmd_args = task.parameters.to_command_args(task.tomo_files, task.output_file)
        for arg, value in cmd_args.items():
            command_parts.append(f"--{arg} {value}")

        for flag in task.parameters.get_flags():
            command_parts.append(f"--{flag}")

        command = " \\\n    ".join(command_parts)
        script_lines.append(command)

        with open(script_path, "w") as f:
            f.write("\n".join(script_lines) + "\n")
        script_path.chmod(0o755)

        print(f"Generated SLURM script: {script_path}")
        return script_path

    def submit_job(self, task) -> str:
        """Submit a single SLURM job."""
        script_path = self.create_sbatch_script(task)

        if self.dry_run:
            return f"DRY_RUN:{script_path}"

        try:
            result = subprocess.run(
                ["sbatch", str(script_path)], capture_output=True, text=True, check=True
            )

            # Parse job ID from sbatch output
            # Typical output: "Submitted batch job 123456"
            job_id = result.stdout.strip().split()[-1]
            print(f"Submitted job {job_id} for {task.tomo_id}")
            return job_id

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to submit {script_path}: {e.stderr}"
            return f"ERROR:{error_msg}"
        except Exception as e:
            error_msg = f"Submission error for {script_path}: {e}"
            return f"ERROR:{error_msg}"

    def submit_jobs(self, tasks: List) -> List[str]:
        """Submit multiple SLURM jobs."""
        job_ids = []
        for task in tasks:
            job_id = self.submit_job(task)
            job_ids.append(job_id)
        return job_ids


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch runner for match_template.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_argument_group("Input Files")
    input_group.add_argument(
        "--tomograms",
        required=True,
        help="Glob pattern for tomogram files (e.g., '/data/tomograms/*.mrc')",
    )
    input_group.add_argument(
        "--metadata",
        required=True,
        help="Glob pattern for metadata files (e.g., '/data/metadata/*.xml')",
    )
    input_group.add_argument(
        "--masks", help="Glob pattern for mask files (e.g., '/data/masks/*.mrc')"
    )
    input_group.add_argument(
        "--template", required=True, type=Path, help="Template file (MRC, PDB, etc.)"
    )
    input_group.add_argument("--template-mask", type=Path, help="Template mask file")
    input_group.add_argument(
        "--tomo-list",
        type=Path,
        help="File with list of tomogram IDs to process (one per line)",
    )

    tm_group = parser.add_argument_group("Template Matching")
    angular_group = tm_group.add_mutually_exclusive_group()
    angular_group.add_argument(
        "--angular-sampling", type=float, help="Angular sampling in degrees"
    )
    angular_group.add_argument(
        "--particle-diameter",
        type=float,
        help="Particle diameter in units of sampling rate (typically Ångstrom)",
    )

    tm_group.add_argument(
        "--score",
        default="FLCSphericalMask",
        help="Template matching scoring function. Use FLC if mask is not spherical.",
    )
    tm_group.add_argument(
        "--score-threshold", type=float, default=0.0, help="Minimum score threshold"
    )

    scope_group = parser.add_argument_group("Microscope Parameters")
    scope_group.add_argument(
        "--voltage", type=float, default=300.0, help="Acceleration voltage in kV"
    )
    scope_group.add_argument(
        "--spherical-aberration",
        type=float,
        default=2.7,
        help="Spherical aberration in mm",
    )
    scope_group.add_argument(
        "--amplitude-contrast", type=float, default=0.07, help="Amplitude contrast"
    )

    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument(
        "--lowpass",
        type=float,
        help="Lowpass filter in units of sampling rate (typically Ångstrom).",
    )
    proc_group.add_argument(
        "--highpass",
        type=float,
        help="Highpass filter in units of sampling rate (typically Ångstrom).",
    )
    proc_group.add_argument(
        "--tilt-weighting",
        choices=["angle", "relion", "grigorieff"],
        help="Tilt weighting scheme",
    )
    proc_group.add_argument(
        "--backend",
        default="cupy",
        choices=list(be._BACKEND_REGISTRY.keys()),
        help="Computation backend",
    )
    proc_group.add_argument(
        "--whiten-spectrum", action="store_true", help="Apply spectral whitening"
    )
    proc_group.add_argument(
        "--scramble-phases",
        action="store_true",
        help="Scramble template phases for noise estimation",
    )

    compute_group = parser.add_argument_group("Compute Resources")
    compute_group.add_argument(
        "--cpus", type=int, default=4, help="Number of CPUs per job"
    )
    compute_group.add_argument(
        "--memory", type=int, default=64, help="Memory per job in GB"
    )
    compute_group.add_argument(
        "--gpu-count", type=int, default=1, help="Number of GPUs per job"
    )
    compute_group.add_argument(
        "--gpu-type", help="GPU type constraint (e.g., '3090', 'A100')"
    )
    compute_group.add_argument(
        "--time-limit", default="05:00:00", help="Time limit (HH:MM:SS)"
    )
    compute_group.add_argument("--partition", default="gpu-el8", help="SLURM partition")

    job_group = parser.add_argument_group("Job Submission")
    job_group.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./batch_results"),
        help="Output directory for results",
    )
    job_group.add_argument(
        "--script-dir",
        type=Path,
        default=Path("./slurm_scripts"),
        help="Directory for generated SLURM scripts",
    )
    job_group.add_argument(
        "--environment-setup",
        default="module load pyTME",
        help="Command(s) to set up pyTME environment",
    )
    job_group.add_argument(
        "--dry-run", action="store_true", help="Generate scripts but do not submit jobs"
    )

    args = parser.parse_args()

    if args.tomo_list is not None:
        with open(args.tomo_list, mode="r") as f:
            args.tomo_list = [line.strip() for line in f if line.strip()]
        print(f"Processing {len(args.tomo_list)} specific tomograms")

    return args


def main():
    print_entry()

    args = parse_args()
    try:
        discovery = TomoDatasetDiscovery(
            mrc_pattern=args.tomograms,
            metadata_pattern=args.metadata,
            mask_pattern=args.masks,
        )
        tomo_files = discovery.discover_tomograms(tomo_list=args.tomo_list)
        print_block(
            name="Discovering Dataset",
            data={
                "Tomogram Pattern": args.tomograms,
                "Metadata Pattern": args.metadata,
                "Mask Pattern": args.masks,
                "Valid Runs": len(tomo_files),
            },
            label_width=30,
        )
        if not tomo_files:
            print("No tomograms found! Check your patterns.")
            return

        params = TMParameters(
            template=args.template,
            template_mask=args.template_mask,
            angular_sampling=args.angular_sampling,
            particle_diameter=args.particle_diameter,
            score=args.score,
            score_threshold=args.score_threshold,
            acceleration_voltage=args.voltage * 1e3,  # keV to eV
            spherical_aberration=args.spherical_aberration * 1e7,  # Convert mm to Å
            amplitude_contrast=args.amplitude_contrast,
            lowpass=args.lowpass,
            highpass=args.highpass,
            tilt_weighting=args.tilt_weighting,
            backend=args.backend,
            whiten_spectrum=args.whiten_spectrum,
            scramble_phases=args.scramble_phases,
        )
        print_params = params.to_command_args(tomo_files[0], "")
        _ = print_params.pop("target")
        _ = print_params.pop("output")
        print_params.update({k: True for k in params.get_flags()})
        print_params = {
            sanitize_name(k): print_params[k] for k in sorted(list(print_params.keys()))
        }
        print_block(name="Matching Parameters", data=print_params, label_width=30)
        print("\n" + "-" * 80)

        resources = ComputeResources(
            cpus=args.cpus,
            memory_gb=args.memory,
            gpu_count=args.gpu_count,
            gpu_type=args.gpu_type,
            time_limit=args.time_limit,
            partition=args.partition,
        )
        print_params = resources.to_slurm_args()
        print_params = {
            sanitize_name(k): print_params[k] for k in sorted(list(print_params.keys()))
        }
        print_block(name="Compute Resources", data=print_params, label_width=30)
        print("\n" + "-" * 80 + "\n")

        tasks = []
        for tomo_file in tomo_files:
            task = TemplateMatchingTask(
                tomo_files=tomo_file,
                parameters=params,
                resources=resources,
                output_dir=args.output_dir,
            )
            tasks.append(task)

        backend = SlurmBackend(
            dry_run=args.dry_run,
            script_dir=args.script_dir,
            environment_setup=args.environment_setup,
        )
        job_ids = backend.submit_jobs(tasks)

        if args.dry_run:
            print(
                f"\nDry run complete. Generated {len(tasks)} scripts in {args.script_dir}"
            )
        else:
            successful_jobs = [j for j in job_ids if not j.startswith("ERROR")]
            print(f"\nSubmitted {len(successful_jobs)} jobs successfully.")
            if successful_jobs:
                print(f"Job IDs:\n{','.join(successful_jobs).strip()}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
