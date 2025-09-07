#!python3
"""
PyTME Batch Runner - Refactored Core Classes
"""
import re
import argparse
import subprocess
from abc import ABC, abstractmethod

from pathlib import Path
from dataclasses import dataclass, field, fields
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


@dataclass
class AnalysisFiles:
    """Container for files related to analysis of a single tomogram."""

    #: Tomogram identifier.
    tomo_id: str
    #: List of TM pickle result files for this tomo_id.
    input_files: List[Path]
    #: Background pickle files for normalization (optional).
    background_files: List[Path] = None
    #: Target mask file (optional).
    mask: Optional[Path] = None

    def __post_init__(self):
        """Validate that required files exist."""
        for input_file in self.input_files:
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")

        if self.background_files:
            for bg_file in self.background_files:
                if not bg_file.exists():
                    raise FileNotFoundError(f"Background file not found: {bg_file}")

        if self.mask and not self.mask.exists():
            raise FileNotFoundError(f"Mask not found: {self.mask}")


class DatasetDiscovery(ABC):
    """Base class for dataset discovery using glob patterns."""

    @abstractmethod
    def discover(self, tomo_list: Optional[List[str]] = None) -> List:
        pass

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

    def create_mapping_table(self, pattern: str) -> Dict[str, List[Path]]:
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

        return ret


@dataclass
class TomoDatasetDiscovery(DatasetDiscovery):
    """Find and match tomogram files using glob patterns."""

    #: Glob pattern for tomogram files, e.g., "/data/tomograms/*.mrc"
    mrc_pattern: str
    #: Glob pattern for metadata files, e.g., "/data/metadata/*.xml"
    metadata_pattern: str
    #: Optional glob pattern for mask files, e.g., "/data/masks/*.mrc"
    mask_pattern: Optional[str] = None

    def discover(self, tomo_list: Optional[List[str]] = None) -> List[TomoFiles]:
        """Find all matching tomogram files."""
        mrc_files = self.create_mapping_table(self.mrc_pattern)
        meta_files = self.create_mapping_table(self.metadata_pattern)
        mask_files = self.create_mapping_table(self.mask_pattern)

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
                    tomogram=value[0].absolute(),
                    metadata=meta_files[key][0].absolute(),
                    mask=mask_files.get(key, [""])[0],
                )
            )
        return tomo_files


@dataclass
class AnalysisDatasetDiscovery(DatasetDiscovery):
    """Find and match analysis files using glob patterns."""

    #: Glob pattern for TM pickle files, e.g., "/data/results/*.pickle"
    input_patterns: List[str]
    #: List of glob patterns for background files, e.g., ["bg1/*.pickle", "bg2/*."]
    background_patterns: List[str] = None
    #: Target masks, e.g., "/data/masks/*.mrc"
    mask_patterns: Optional[str] = None

    def __post_init__(self):
        """Ensure patterns are lists."""
        if isinstance(self.input_patterns, str):
            self.input_patterns = [self.input_patterns]
        if self.background_patterns and isinstance(self.background_patterns, str):
            self.background_patterns = [self.background_patterns]

    def discover(self, tomo_list: Optional[List[str]] = None) -> List[AnalysisFiles]:
        """Find all matching analysis files."""

        input_files_by_id = {}
        for pattern in self.input_patterns:
            files = self.create_mapping_table(pattern)
            for tomo_id, file_list in files.items():
                if tomo_id not in input_files_by_id:
                    input_files_by_id[tomo_id] = []
                input_files_by_id[tomo_id].extend(file_list)

        background_files_by_id = {}
        if self.background_patterns:
            for pattern in self.background_patterns:
                bg_files = self.create_mapping_table(pattern)
                for tomo_id, file_list in bg_files.items():
                    if tomo_id not in background_files_by_id:
                        background_files_by_id[tomo_id] = []
                    background_files_by_id[tomo_id].extend(file_list)

        mask_files_by_id = {}
        if self.mask_patterns:
            mask_files_by_id = self.create_mapping_table(self.mask_patterns)

        if tomo_list:
            input_files_by_id = {
                k: v for k, v in input_files_by_id.items() if k in tomo_list
            }
            background_files_by_id = {
                k: v for k, v in background_files_by_id.items() if k in tomo_list
            }
            mask_files_by_id = {
                k: v for k, v in mask_files_by_id.items() if k in tomo_list
            }

        analysis_files = []
        for tomo_id, input_file_list in input_files_by_id.items():
            background_files = background_files_by_id.get(tomo_id, [])
            mask_file = mask_files_by_id.get(tomo_id, [None])[0]

            analysis_file = AnalysisFiles(
                tomo_id=tomo_id,
                input_files=[f.absolute() for f in input_file_list],
                background_files=(
                    [f.absolute() for f in background_files] if background_files else []
                ),
                mask=mask_file.absolute() if mask_file else None,
            )
            analysis_files.append(analysis_file)

        return analysis_files


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
    invert_cone: bool = field(default=False, metadata={"flag": True})
    no_use_optimized_set: bool = field(default=False, metadata={"flag": True})

    # Microscope parameters
    acceleration_voltage: float = 300.0  # kV
    spherical_aberration: float = 2.7e7  # Å
    amplitude_contrast: float = 0.07
    defocus: Optional[float] = None  # Å
    phase_shift: float = 0.0  # degrees

    # Processing options
    lowpass: Optional[float] = None  # Å
    highpass: Optional[float] = None  # Å
    pass_format: str = "sampling_rate"  # "sampling_rate", "voxel", "frequency"
    no_pass_smooth: bool = field(default=True, metadata={"flag": False})
    interpolation_order: int = 3
    score_threshold: float = 0.0
    score: str = "FLCSphericalMask"
    background_correction: Optional[str] = None

    # Weighting and correction
    tilt_weighting: Optional[str] = None  # "angle", "relion", "grigorieff"
    wedge_axes: str = "2,0"
    whiten_spectrum: bool = field(default=False, metadata={"flag": True})
    scramble_phases: bool = field(default=False, metadata={"flag": True})
    invert_target_contrast: bool = field(default=False, metadata={"flag": True})

    # CTF parameters
    ctf_file: Optional[Path] = None
    no_flip_phase: bool = field(default=True, metadata={"flag": False})
    correct_defocus_gradient: bool = field(default=False, metadata={"flag": True})

    # Performance options
    centering: bool = field(default=False, metadata={"flag": True})
    pad_edges: bool = field(default=False, metadata={"flag": True})
    pad_filter: bool = field(default=False, metadata={"flag": True})
    use_mixed_precision: bool = field(default=False, metadata={"flag": True})
    use_memmap: bool = field(default=False, metadata={"flag": True})

    # Analysis options
    peak_calling: bool = field(default=False, metadata={"flag": True})
    num_peaks: int = 1000

    # Backend selection
    backend: str = "numpy"
    gpu_indices: Optional[str] = None

    # Reconstruction
    reconstruction_filter: str = "ramp"
    reconstruction_interpolation_order: int = 1
    no_filter_target: bool = field(default=False, metadata={"flag": True})

    def __post_init__(self):
        """Validate parameters and convert units."""
        self.template = self.template.absolute()
        if self.template_mask:
            self.template_mask = self.template_mask.absolute()

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

    def to_command_args(self, files: TomoFiles, output_path: Path) -> Dict[str, Any]:
        """Convert parameters to pyTME command arguments."""
        args = {
            "target": str(files.tomogram),
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
        if files.mask:
            args["target-mask"] = str(files.mask)
        if files.metadata:
            args["ctf-file"] = str(files.metadata)
            args["tilt-angles"] = str(files.metadata)

        if self.background_correction:
            args["background-correction"] = self.background_correction

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
        return {k: v for k, v in args.items() if v is not None}

    def get_flags(self) -> List[str]:
        flags = []

        for field_info in fields(self):
            flag_meta = field_info.metadata.get("flag")
            if flag_meta is None:
                continue

            value = getattr(self, field_info.name)
            if not isinstance(value, bool):
                continue

            flag_name = field_info.name.replace("_", "-")
            if (flag_meta is True and value) or (flag_meta is False and not value):
                flags.append(flag_name)

        return flags


@dataclass
class AnalysisParameters(TMParameters):
    """Parameters for template matching analysis and peak calling."""

    # Peak calling
    peak_caller: str = "PeakCallerMaximumFilter"
    num_peaks: int = 1000
    min_score: float = 0.0
    max_score: Optional[float] = None
    min_distance: int = 5
    min_boundary_distance: int = 0
    mask_edges: bool = field(default=False, metadata={"flag": True})
    n_false_positives: Optional[int] = None

    # Output format
    output_format: str = "relion4"
    output_directory: Optional[str] = None

    # Advanced options
    extraction_box_size: Optional[int] = None

    def to_command_args(
        self, files: AnalysisFiles, output_path: Path
    ) -> Dict[str, Any]:
        """Convert parameters to analyze_template_matching command arguments."""
        args = {
            "input-files": " ".join([str(f) for f in files.input_files]),
            "output-prefix": str(output_path.parent / output_path.stem),
            "peak-caller": self.peak_caller,
            "num-peaks": self.num_peaks,
            "min-score": self.min_score,
            "min-distance": self.min_distance,
            "min-boundary-distance": self.min_boundary_distance,
            "output-format": self.output_format,
        }

        # Optional parameters
        if self.max_score is not None:
            args["max-score"] = self.max_score
        if self.n_false_positives is not None:
            args["n-false-positives"] = self.n_false_positives
        if self.extraction_box_size is not None:
            args["extraction-box-size"] = self.extraction_box_size
        if files.mask:
            args["target-mask"] = str(files.mask)

        # Background files
        if files.background_files:
            args["background-files"] = " ".join(
                [str(f) for f in files.background_files]
            )

        return {k: v for k, v in args.items() if v is not None}


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
class AbstractTask(ABC):
    """Abstract task specification"""

    files: object
    parameters: object
    resources: ComputeResources
    output_dir: Path

    @property
    def tomo_id(self) -> str:
        return self.files.tomo_id

    @abstractmethod
    def executable(self) -> str:
        pass

    @property
    @abstractmethod
    def output_file(self) -> Path:
        pass

    def to_command_args(self):
        return self.parameters.to_command_args(self.files, self.output_file)

    def create_output_dir(self) -> None:
        """Ensure output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TemplateMatchingTask(AbstractTask):
    """Template matching task."""

    @property
    def output_file(self) -> Path:
        original_stem = self.files.tomogram.stem
        return self.output_dir / f"{original_stem}.pickle"

    @property
    def executable(self):
        return "match_template"


class AnalysisTask(AbstractTask):
    """Analysis task for processing TM results."""

    @property
    def output_file(self) -> Path:
        """Generate output filename based on format."""
        prefix = self.files.input_files[0].stem

        format_extensions = {
            "orientations": ".tsv",
            "relion4": ".star",
            "relion5": ".star",
            "pickle": ".pickle",
            "alignment": "",
            "extraction": "",
            "average": ".mrc",
        }

        extension = format_extensions.get(self.parameters.output_format, ".tsv")
        return self.output_dir / f"{prefix}{extension}"

    @property
    def executable(self):
        return "postprocess"


class ExecutionBackend(ABC):
    """Abstract base class for execution backends."""

    @abstractmethod
    def submit_job(self, task) -> str:
        """Submit a single job and return job ID or status."""

    @abstractmethod
    def submit_jobs(self, tasks: List) -> List[str]:
        """Submit multiple jobs and return list of job IDs."""


class SlurmBackend(ExecutionBackend):
    """SLURM execution backend for cluster job submission."""

    def __init__(
        self,
        force: bool = True,
        dry_run: bool = False,
        script_dir: Path = Path("./slurm_scripts"),
        environment_setup: str = "module load pyTME",
    ):
        """
        Initialize SLURM backend.

        Parameters
        ----------
        force : bool, optional
            Rerun completed jobs, defaults to True.
        dry_run : bool, optional
            Generate scripts but do not submit, defaults to False.
        script_dir: str, optional
            Directory to save generated scripts, defaults to ./slurm_scripts,
        environment_setup : str, optional
            Command to set up pyTME environment, defaults to module load pyTME.
        """
        self.force = force
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
                "job-name": f"pytme_{task.executable}_{task.tomo_id}",
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

        command_parts = [task.executable]
        cmd_args = task.to_command_args()
        for arg, value in cmd_args.items():
            command_parts.append(f"--{arg} {value}")

        for flag in task.parameters.get_flags():
            command_parts.append(f"--{flag}")

        command = " \\\n    ".join(command_parts)
        script_lines.append(command)

        with open(script_path, "w") as f:
            f.write("\n".join(script_lines) + "\n")
        script_path.chmod(0o755)

        print(f"Generated SLURM script: {Path(script_path).name}")
        return script_path

    def submit_job(self, task) -> str:
        """Submit a single SLURM job."""
        script_path = self.create_sbatch_script(task)

        if self.dry_run:
            return f"DRY_RUN:{script_path}"

        try:
            if Path(task.output_file).exists() and not self.force:
                return f"ERROR: {str(task.output_file)} exists and force was not set."

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


def add_compute_resources(
    parser,
    default_cpus=4,
    default_memory=32,
    default_time="02:00:00",
    default_partition="cpu",
    include_gpu=False,
):
    """Add compute resource arguments to a parser."""
    compute_group = parser.add_argument_group("Compute Resources")
    compute_group.add_argument(
        "--cpus", type=int, default=default_cpus, help="Number of CPUs per job"
    )
    compute_group.add_argument(
        "--memory", type=int, default=default_memory, help="Memory per job in GB"
    )
    compute_group.add_argument(
        "--time-limit", default=default_time, help="Time limit (HH:MM:SS)"
    )
    compute_group.add_argument(
        "--partition", default=default_partition, help="SLURM partition"
    )
    compute_group.add_argument(
        "--qos", default="normal", help="SLURM quality of service"
    )

    if include_gpu:
        compute_group.add_argument(
            "--gpu-count", type=int, default=1, help="Number of GPUs per job"
        )
        compute_group.add_argument(
            "--gpu-type",
            default="3090",
            help="GPU type constraint (e.g., '3090', 'A100')",
        )

    return compute_group


def add_job_submission(parser, default_output_dir="./results"):
    """Add job submission arguments to a parser."""
    job_group = parser.add_argument_group("Job Submission")
    job_group.add_argument(
        "--output-dir",
        type=Path,
        default=Path(default_output_dir),
        help="Output directory for results",
    )
    job_group.add_argument(
        "--script-dir",
        type=Path,
        default=Path("./scripts"),
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
    job_group.add_argument("--force", action="store_true", help="Rerun completed jobs")

    return job_group


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch runner for PyTME.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    matching_parser = subparsers.add_parser(
        "matching",
        help="Run template matching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input files for matching
    tm_input_group = matching_parser.add_argument_group("Input Files")
    tm_input_group.add_argument(
        "--tomograms",
        required=True,
        help="Glob pattern for tomogram files (e.g., '/data/tomograms/*.mrc')",
    )
    tm_input_group.add_argument(
        "--metadata",
        required=True,
        help="Glob pattern for metadata files (e.g., '/data/metadata/*.xml')",
    )
    tm_input_group.add_argument(
        "--masks", help="Glob pattern for target mask files (e.g., '/data/masks/*.mrc')"
    )
    tm_input_group.add_argument(
        "--template", required=True, type=Path, help="Template file (MRC, PDB, etc.)"
    )
    tm_input_group.add_argument("--template-mask", type=Path, help="Template mask file")
    tm_input_group.add_argument(
        "--tomo-list",
        type=Path,
        help="File with list of tomogram IDs to process (one per line)",
    )

    # Template matching parameters
    tm_group = matching_parser.add_argument_group("Template Matching")
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
        "--background-correction",
        choices=["phase-scrambling"],
        required=False,
        help="Transform cross-correlation into SNR-like values using a given method: "
        "'phase-scrambling' uses a phase-scrambled template as background",
    )
    tm_group.add_argument(
        "--score-threshold", type=float, default=0.0, help="Minimum score threshold"
    )

    # Microscope parameters
    scope_group = matching_parser.add_argument_group("Microscope Parameters")
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

    # Processing options
    proc_group = matching_parser.add_argument_group("Processing Options")
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

    _ = add_compute_resources(
        matching_parser,
        default_cpus=4,
        default_memory=64,
        include_gpu=True,
        default_time="05:00:00",
        default_partition="gpu-el8",
    )
    _ = add_job_submission(matching_parser, "./matching_results")

    analysis_parser = subparsers.add_parser(
        "analysis",
        help="Analyze template matching results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input files for analysis
    analysis_input_group = analysis_parser.add_argument_group("Input Files")
    analysis_input_group.add_argument(
        "--input-file",
        "--input-files",
        required=True,
        nargs="+",
        help="Path to one or multiple runs of match_template.py.",
    )
    analysis_input_group.add_argument(
        "--background-file",
        "--background-files",
        required=False,
        nargs="+",
        default=[],
        help="Path to one or multiple runs of match_template.py for normalization. "
        "For instance from --scramble_phases or a different template.",
    )
    analysis_input_group.add_argument(
        "--masks", help="Glob pattern for target mask files (e.g., '/data/masks/*.mrc')"
    )
    analysis_input_group.add_argument(
        "--tomo-list",
        type=Path,
        help="File with list of tomogram IDs to process (one per line)",
    )

    # Peak calling parameters
    peak_group = analysis_parser.add_argument_group("Peak Calling")
    peak_group.add_argument(
        "--peak-caller",
        choices=[
            "PeakCallerSort",
            "PeakCallerMaximumFilter",
            "PeakCallerFast",
            "PeakCallerRecursiveMasking",
            "PeakCallerScipy",
        ],
        default="PeakCallerMaximumFilter",
        help="Peak caller for local maxima identification",
    )
    peak_group.add_argument(
        "--num-peaks",
        type=int,
        default=1000,
        help="Maximum number of peaks to identify",
    )
    peak_group.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Minimum score from which peaks will be considered",
    )
    peak_group.add_argument(
        "--max-score",
        type=float,
        default=None,
        help="Maximum score until which peaks will be considered",
    )
    peak_group.add_argument(
        "--min-distance", type=int, default=None, help="Minimum distance between peaks"
    )
    peak_group.add_argument(
        "--min-boundary-distance",
        type=int,
        default=None,
        help="Minimum distance of peaks to target edges",
    )
    peak_group.add_argument(
        "--mask-edges",
        action="store_true",
        default=False,
        help="Whether candidates should not be identified from scores that were "
        "computed from padded densities. Superseded by min_boundary_distance.",
    )
    peak_group.add_argument(
        "--n-false-positives",
        type=int,
        default=None,
        help="Number of accepted false-positive picks to determine minimum score",
    )

    # Output options
    output_group = analysis_parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-format",
        choices=[
            "orientations",
            "relion4",
            "relion5",
            "alignment",
            "extraction",
            "average",
            "pickle",
        ],
        default="relion4",
        help="Output format for analysis results",
    )

    advanced_group = analysis_parser.add_argument_group("Advanced Options")
    advanced_group.add_argument(
        "--extraction-box-size",
        type=int,
        default=None,
        help="Box size for extracted subtomograms (for extraction output format)",
    )

    _ = add_compute_resources(
        analysis_parser,
        default_cpus=2,
        default_memory=16,
        include_gpu=False,
        default_time="01:00:00",
        default_partition="htc-el8",
    )
    _ = add_job_submission(analysis_parser, "./analysis_results")

    args = parser.parse_args()
    if args.tomo_list is not None:
        with open(args.tomo_list, mode="r") as f:
            args.tomo_list = [line.strip() for line in f if line.strip()]

    args.output_dir = args.output_dir.absolute()
    args.script_dir = args.script_dir.absolute()
    return args


def run_matching(args, resources):
    discovery = TomoDatasetDiscovery(
        mrc_pattern=args.tomograms,
        metadata_pattern=args.metadata,
        mask_pattern=args.masks,
    )
    files = discovery.discover(tomo_list=args.tomo_list)
    print_block(
        name="Discovering Dataset",
        data={
            "Tomogram Pattern": args.tomograms,
            "Metadata Pattern": args.metadata,
            "Mask Pattern": args.masks,
            "Valid Runs": len(files),
        },
        label_width=30,
    )
    if not files:
        print("No tomograms found! Check your patterns.")
        return

    params = TMParameters(
        template=args.template,
        template_mask=args.template_mask,
        angular_sampling=args.angular_sampling,
        particle_diameter=args.particle_diameter,
        score=args.score,
        score_threshold=args.score_threshold,
        acceleration_voltage=args.voltage,
        spherical_aberration=args.spherical_aberration * 1e7,  # mm to Ångstrom
        amplitude_contrast=args.amplitude_contrast,
        lowpass=args.lowpass,
        highpass=args.highpass,
        tilt_weighting=args.tilt_weighting,
        backend=args.backend,
        whiten_spectrum=args.whiten_spectrum,
        scramble_phases=args.scramble_phases,
        background_correction=args.background_correction,
    )
    print_params = params.to_command_args(files[0], "")
    _ = print_params.pop("target")
    _ = print_params.pop("output")
    print_params.update({k: True for k in params.get_flags()})
    print_params = {
        sanitize_name(k): print_params[k] for k in sorted(list(print_params.keys()))
    }
    print_block(name="Matching Parameters", data=print_params, label_width=30)
    print("\n" + "-" * 80)

    tasks = []
    for tomo_file in files:
        task = TemplateMatchingTask(
            files=tomo_file,
            parameters=params,
            resources=resources,
            output_dir=args.output_dir,
        )
        tasks.append(task)

    return tasks


def run_analysis(args, resources):
    discovery = AnalysisDatasetDiscovery(
        input_patterns=args.input_file,
        background_patterns=args.background_file,
        mask_patterns=args.masks,
    )
    files = discovery.discover(tomo_list=args.tomo_list)
    print_block(
        name="Discovering Dataset",
        data={
            "Input Patterns": args.input_file,
            "Background Patterns": args.background_file,
            "Mask Pattern": args.masks,
            "Valid Runs": len(files),
        },
        label_width=30,
    )
    if not files:
        print("No TM results found! Check your patterns.")
        return

    params = AnalysisParameters(
        peak_caller=args.peak_caller,
        num_peaks=args.num_peaks,
        min_score=args.min_score,
        max_score=args.max_score,
        min_distance=args.min_distance,
        min_boundary_distance=args.min_boundary_distance,
        mask_edges=args.mask_edges,
        n_false_positives=args.n_false_positives,
        output_format=args.output_format,
        extraction_box_size=args.extraction_box_size,
    )
    print_params = params.to_command_args(files[0], Path(""))
    _ = print_params.pop("input-files", None)
    _ = print_params.pop("background-files", None)
    _ = print_params.pop("output-prefix", None)
    print_params.update({k: True for k in params.get_flags()})
    print_params = {
        sanitize_name(k): print_params[k] for k in sorted(list(print_params.keys()))
    }
    print_block(name="Analysis Parameters", data=print_params, label_width=30)
    print("\n" + "-" * 80)

    tasks = []
    for file in files:
        task = AnalysisTask(
            files=file,
            parameters=params,
            resources=resources,
            output_dir=args.output_dir,
        )
        tasks.append(task)

    return tasks


def main():
    print_entry()

    args = parse_args()

    resources = ComputeResources(
        cpus=args.cpus,
        memory_gb=args.memory,
        time_limit=args.time_limit,
        partition=args.partition,
        gpu_count=getattr(args, "gpu_count", 0),
        gpu_type=getattr(args, "gpu_type", None),
    )

    func = run_matching
    if args.command == "analysis":
        func = run_analysis

    try:
        tasks = func(args, resources)
    except Exception as e:
        exit(f"Error: {e}")

    if tasks is None:
        exit(-1)

    print_params = resources.to_slurm_args()
    print_params = {
        sanitize_name(k): print_params[k] for k in sorted(list(print_params.keys()))
    }
    print_block(name="Compute Resources", data=print_params, label_width=30)
    print("\n" + "-" * 80 + "\n")

    backend = SlurmBackend(
        force=args.force,
        dry_run=args.dry_run,
        script_dir=args.script_dir,
        environment_setup=args.environment_setup,
    )
    job_ids = backend.submit_jobs(tasks)
    if args.dry_run:
        print(
            f"\nDry run complete. Generated {len(tasks)} scripts in {args.script_dir}"
        )
        return 0

    successful_jobs = [j for j in job_ids if not j.startswith("ERROR")]
    print(f"\nSubmitted {len(successful_jobs)} jobs successfully.")
    if successful_jobs:
        print(f"Job IDs:\n{','.join(successful_jobs).strip()}")

    if len(successful_jobs) == len(job_ids):
        return 0

    print("\nThe following issues arose during submission:")
    for j in job_ids:
        if j.startswith("ERROR"):
            print(j)


if __name__ == "__main__":
    main()
