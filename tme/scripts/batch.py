#!python3
"""
Pytme batch executor aiming to simplify working with large datasets.

Copyright (c) 2025 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import re
import shlex
import argparse
import subprocess
from tempfile import gettempdir
from abc import ABC, abstractmethod

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set, Tuple

from tme.utils import cli, logging
from tme.scripts.match_template import add_matching_arguments
from tme.scripts.postprocess import add_postprocess_arguments

logger = logging.get_logger("batch")


@dataclass
class TomoFiles:
    """Container for all files related to a single tomogram."""

    #: Tomogram identifier.
    tomo_id: str
    #: Path to tomogram.
    tomogram: Path
    #: XML file with tilt angles, defocus, etc, optional.
    metadata: Optional[Path]
    #: Path to tomogram mask, optional.
    mask: Optional[Path] = None
    #: Path to seed points for constrained matching, optional.
    orientations: Optional[Path] = None

    def __post_init__(self):
        """Validate that required files exist."""
        if not self.tomogram.exists():
            raise FileNotFoundError(f"Tomogram not found: {self.tomogram}")
        if self.metadata and not self.metadata.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata}")
        if self.mask and not self.mask.exists():
            raise FileNotFoundError(f"Mask not found: {self.mask}")
        if self.orientations and not self.orientations.exists():
            raise FileNotFoundError(f"Orientations not found: {self.orientations}")


@dataclass
class AnalysisFiles:
    """Container for files related to analysis of a single tomogram."""

    #: Tomogram identifier.
    tomo_id: str
    #: List of TM pickle result files for this tomo_id.
    input_files: List[Path]
    #: Background pickle files for normalization (optional).
    background_files: Optional[List[Path]] = None
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


@dataclass
class OptimizeFiles:
    """Container for a batch of optimize indices."""

    tomo_id: str
    manifest: Path
    output_prefix: str
    indices: List[int]

    def __post_init__(self):
        if not self.manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest}")


class DatasetDiscovery(ABC):
    """Base class for dataset discovery using glob patterns."""

    _MISSING_STR = "\033[91mNo\033[0m"

    @abstractmethod
    def discover(self, tomo_list: Optional[List[str]] = None) -> List:
        pass

    @staticmethod
    def safe_get(mapping, key, default=None):
        try:
            return mapping.get(key)[0].absolute()
        except Exception:
            return default

    @staticmethod
    def parse_id_from_filename(filename: str) -> str:
        """Extract the tomogram ID from filename by removing technical suffixes."""
        base = cli.strip_result_extensions(filename)
        # Remove technical suffixes (pixel size, binning, filtering info)
        # Examples: "_10.00Apx", "_4.00Apx", "_bin4", "_dose_filt"
        base = re.sub(r"_\d+(\.\d+)?(Apx|bin\d*|dose_filt)$", "", base)

        # Remove common organizational prefixes if they exist
        for prefix in ["rec_Position_", "Position_", "rec_", "tomo_"]:
            if base.startswith(prefix):
                base = base[len(prefix) :]
                break
        return base

    @staticmethod
    def _filter_by_tomo_list(mapping, tomo_list):
        if not tomo_list:
            return mapping
        tomo_set = set(tomo_list)
        return {k: v for k, v in mapping.items() if k in tomo_set}

    @classmethod
    def _log_discovery(cls, tomo_id, header, entries):
        """Log a formatted discovery line.

        Parameters
        ----------
        tomo_id : str
            Tomogram identifier.
        header : str
            Initial status text, e.g. ``"tomo Ok"`` or ``"inputs 3"``.
        entries : list of (label, is_found, pattern_provided)
            Each entry adds ``" | label Ok"`` or ``" | label No"`` when
            *pattern_provided* is truthy.
        """
        strout = f"{tomo_id}: {header}"
        for label, is_found, pattern_provided in entries:
            if pattern_provided:
                status = "Ok" if is_found else cls._MISSING_STR
                strout += f" | {label} {status}"
        logger.info(strout)

    def create_mapping_table(self, pattern: Optional[str]) -> Dict[str, List[Path]]:
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
    #: Optional glob pattern for metadata files, e.g., "/data/metadata/*.xml"
    metadata_pattern: Optional[str] = None
    #: Optional glob pattern for mask files, e.g., "/data/masks/*.mrc"
    mask_pattern: Optional[str] = None
    #: Optional glob pattern for seed points, e.g., "/data/seed_points/*.star"
    orientation_pattern: Optional[str] = None
    #: Raise an error if not all provided patterns yield results for all tomograms.
    strict: bool = False

    def discover(self, tomo_list: Optional[List[str]] = None) -> List[TomoFiles]:
        """Find all matching tomogram files."""
        mrc_files = self.create_mapping_table(self.mrc_pattern)
        meta_files = self.create_mapping_table(self.metadata_pattern)
        mask_files = self.create_mapping_table(self.mask_pattern)
        orientation_files = self.create_mapping_table(self.orientation_pattern)

        mrc_files = self._filter_by_tomo_list(mrc_files, tomo_list)
        meta_files = self._filter_by_tomo_list(meta_files, tomo_list)
        mask_files = self._filter_by_tomo_list(mask_files, tomo_list)
        orientation_files = self._filter_by_tomo_list(orientation_files, tomo_list)
        strict_errors, tomo_files = [], []
        for key in sorted(list(mrc_files.keys())):
            value = mrc_files[key]

            entries = [
                ("metadata", key in meta_files, self.metadata_pattern),
                ("mask", key in mask_files, self.mask_pattern),
                ("orientations", key in orientation_files, self.orientation_pattern),
            ]
            self._log_discovery(key, "tomo Ok", entries)

            if self.strict:
                missing = [
                    label
                    for label, is_found, pattern in entries
                    if not is_found and pattern
                ]

                if missing:
                    strict_errors.append(f"  - {key}: missing {', '.join(missing)}")

            tomo_files.append(
                TomoFiles(
                    tomo_id=key,
                    tomogram=value[0].absolute(),
                    metadata=self.safe_get(meta_files, key),
                    mask=self.safe_get(mask_files, key),
                    orientations=self.safe_get(orientation_files, key),
                )
            )

        if self.strict and strict_errors:
            error_msg = "Strict mode enabled but files are missing:\n" + "\n".join(
                strict_errors
            )
            raise ValueError(error_msg)
        return tomo_files


@dataclass
class AnalysisDatasetDiscovery(DatasetDiscovery):
    """Find and match analysis files using glob patterns."""

    #: Glob pattern for TM pickle files, e.g., "/data/results/*.pickle"
    input_patterns: List[str]
    #: List of glob patterns for background files, e.g., ["bg1/*.pickle", "bg2/*."]
    background_patterns: Optional[List[str]] = None
    #: Target masks, e.g., "/data/masks/*.mrc"
    mask_patterns: Optional[str] = None
    #: Raise an error if not all provided patterns yield results for all inputs.
    strict: bool = False

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

        input_files_by_id = self._filter_by_tomo_list(input_files_by_id, tomo_list)
        background_files_by_id = self._filter_by_tomo_list(
            background_files_by_id, tomo_list
        )
        mask_files_by_id = self._filter_by_tomo_list(mask_files_by_id, tomo_list)
        strict_errors, analysis_files = [], []
        for tomo_id, input_file_list in sorted(input_files_by_id.items()):
            background_files = background_files_by_id.get(tomo_id, [])
            mask_file = mask_files_by_id.get(tomo_id, [None])[0]

            entries = [
                ("background", bool(background_files), self.background_patterns),
                ("mask", mask_file is not None, self.mask_patterns),
            ]
            self._log_discovery(tomo_id, f"inputs {len(input_file_list)}", entries)

            if self.strict:
                missing = [
                    label
                    for label, is_found, pattern in entries
                    if not is_found and pattern
                ]
                if missing:
                    strict_errors.append(f"  - {tomo_id}: missing {', '.join(missing)}")

            analysis_file = AnalysisFiles(
                tomo_id=tomo_id,
                input_files=[f.absolute() for f in input_file_list],
                background_files=(
                    [f.absolute() for f in background_files] if background_files else []
                ),
                mask=mask_file.absolute() if mask_file else None,
            )
            analysis_files.append(analysis_file)

        if self.strict and strict_errors:
            error_msg = "Strict mode enabled but files are missing:\n" + "\n".join(
                strict_errors
            )
            raise ValueError(error_msg)
        return analysis_files


def args_to_command_dict(
    args: argparse.Namespace,
    keep: Set[str],
    flags: Set[str],
) -> Dict[str, Any]:
    result = {}
    for key, value in vars(args).items():
        if key not in keep or value is None:
            continue

        cli_key = f"--{key.replace('_', '-')}"

        if cli_key in flags:
            if value is True:
                # None signals this is a flag not a value
                result[cli_key] = None
            continue

        if isinstance(value, bool):
            continue

        if isinstance(value, Path):
            value = str(value)

        if isinstance(value, (list, tuple)):
            value = " ".join(str(v) for v in value)

        result[cli_key] = value

    return result


@dataclass
class _TaskBase(ABC):
    """Base class for batch tasks."""

    files: Any
    args: argparse.Namespace
    output_dir: Path
    cli_args: cli.ArgumentMetadata

    @property
    def tomo_id(self) -> str:
        return self.files.tomo_id

    @property
    @abstractmethod
    def executable(self) -> str: ...

    @property
    @abstractmethod
    def output_file(self) -> Path: ...

    @abstractmethod
    def to_command_args(self) -> Dict[str, Any]: ...

    def create_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TemplateMatchingTask(_TaskBase):
    files: TomoFiles

    @property
    def executable(self) -> str:
        return "pytme match"

    @property
    def output_file(self) -> Path:
        ext = getattr(self.args, "output_format", "pickle")
        if ext is None:
            ext = "pickle"
        return self.output_dir / f"{self.files.tomogram.stem}.{ext}"

    def to_command_args(self) -> Dict[str, Any]:
        cmd_args = {
            "--target": str(self.files.tomogram),
            "--output": str(self.output_file),
        }

        if self.files.mask:
            cmd_args["--target-mask"] = str(self.files.mask)
        if self.files.metadata:
            cmd_args["--ctf-file"] = str(self.files.metadata)
            cmd_args["--tilt-angles"] = str(self.files.metadata)
        if self.files.orientations:
            cmd_args["--orientations"] = str(self.files.orientations)

        cmd_args.update(
            args_to_command_dict(self.args, self.cli_args.args, self.cli_args.flags)
        )
        return cmd_args


@dataclass
class AnalysisTask(_TaskBase):
    files: AnalysisFiles

    @property
    def executable(self) -> str:
        return "pytme postprocess"

    @property
    def output_prefix(self) -> Path:
        prefix = cli.strip_result_extensions(self.files.input_files[0].name)
        return self.output_dir / prefix

    @property
    def output_file(self) -> Path:
        # All of this is done to check whether the files created by
        # postprocess already exist or not
        format_extensions = {
            "orientations": ".tsv",
            "relion4": ".star",
            "relion5": ".star",
            "pickle": ".pickle",
            "alignment": "",
            "extraction": "",
            "average": ".mrc",
        }
        ext = getattr(self.args, "output_format", "relion4")
        extension = format_extensions.get(ext, ".tsv")
        return Path(f"{self.output_prefix}{extension}")

    def to_command_args(self) -> Dict[str, Any]:
        cmd_args = {
            "--input-files": " ".join([str(f) for f in self.files.input_files]),
            "--output-prefix": str(self.output_prefix),
        }

        if self.files.mask:
            cmd_args["--target-mask"] = str(self.files.mask)
        if self.files.background_files:
            cmd_args["--background-files"] = " ".join(
                [str(f) for f in self.files.background_files]
            )

        cmd_args.update(
            args_to_command_dict(self.args, self.cli_args.args, self.cli_args.flags)
        )
        return cmd_args


@dataclass
class OptimizeTask:
    """Task for running pytme optimize on a subset of matching indices."""

    files: OptimizeFiles
    args: argparse.Namespace
    output_dir: Path

    @property
    def tomo_id(self) -> str:
        return self.files.tomo_id

    @property
    def executable(self) -> str:
        return "pytme optimize"

    @property
    def output_file(self) -> Path:
        return self.output_dir / f"{self.tomo_id}.yaml"

    def to_command_args(self) -> Dict[str, Any]:
        cmd_args = {
            "--manifest": str(self.files.manifest),
            "--output-prefix": self.files.output_prefix,
            "--index": " ".join(str(i) for i in self.files.indices),
        }
        if getattr(self.args, "no_match", False):
            cmd_args["--no-match"] = None
        if getattr(self.args, "no_postprocess", False):
            cmd_args["--no-postprocess"] = None
        return cmd_args

    def create_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


class ExecutionBackend:
    """Generic script-based job executor."""

    def __init__(
        self,
        submit_command: str = "sbatch",
        submit_args: str = "",
        script_dir: Optional[Path] = None,
        dry_run: bool = False,
        force: bool = True,
        environment_setup: str = "",
    ):
        """
        Initialize script executor.

        Parameters
        ----------
        submit_command : str
            Command to submit scripts (e.g., 'sbatch', 'qsub', 'bsub', 'bash')
        submit_args : str
            Arguments passed to submit command.
        script_dir : Path
            Directory to save generated scripts
        dry_run : bool
            Generate scripts but do not submit
        force : bool
            Rerun completed jobs
        environment_setup : str
            Command(s) to set up environment
        """
        if script_dir is None:
            script_dir = Path(gettempdir())
        self.script_dir = script_dir
        self.script_dir.mkdir(exist_ok=True, parents=True)

        self.force = force
        self.dry_run = dry_run

        self.submit_args = submit_args
        self.submit_command = submit_command
        self.environment_setup = environment_setup

    def create_script(self, task, path, submit_command: str = ""):
        """Generate execution script for a task."""
        task.create_output_dir()
        command_parts = [task.executable]
        cmd_args = task.to_command_args()
        for arg, value in cmd_args.items():
            if value is None:
                command_parts.append(arg)  # Flag without value
            else:
                command_parts.append(f"{arg} {value}")

        script_lines = ["#!/bin/bash", ""]
        if self.environment_setup:
            script_lines.extend(["# Environment setup", self.environment_setup, ""])

        if submit_command:
            script_lines.extend(["# Submission command", f"# {submit_command}", ""])

        script_lines.append(" \\\n    ".join(command_parts))
        with open(path, "w") as f:
            f.write("\n".join(script_lines) + "\n")
        path.chmod(0o755)

    def submit_job(self, task) -> Tuple[bool, str]:
        """Submit a single job."""

        script_path = None
        try:
            output_file = task.output_file
            if Path(output_file).exists() and not self.dry_run and not self.force:
                return False, f"{str(output_file)} exists and force was not set."

            script_path = self.script_dir / f"pytme_{task.tomo_id}.sh"

            cmd = [self.submit_command]
            if self.submit_args:
                placeholders = {
                    "output_file": f"{task.output_dir}/{task.tomo_id}_%j.out",
                    "error_file": f"{task.output_dir}/{task.tomo_id}_%j.err",
                    "job_name": f"{task.executable.replace(' ', '_')}_{task.tomo_id}",
                }
                cmd.extend(shlex.split(self.submit_args.format(**placeholders)))
            cmd.append(str(script_path))

            self.create_script(task, path=script_path, submit_command=shlex.join(cmd))

            if self.dry_run:
                if Path(output_file).exists():
                    return False, f"DRY_RUN: {output_file} already exists, skipping."
                return True, f"DRY_RUN: {script_path}"

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            job_id = self._parse_job_id(result.stdout)
            logger.info(f"Submitted job {job_id} for {task.tomo_id}")
            return True, str(job_id)

        except subprocess.CalledProcessError as e:
            return False, f"Failed to submit {script_path or task.tomo_id}: {e.stderr}"
        except Exception as e:
            return False, f"Submission error for {script_path or task.tomo_id}: {e}"

    def _parse_job_id(self, stdout: str) -> str:
        """Parse job ID from submission output."""
        return stdout.strip().split()[-1] if stdout.strip() else "SUBMITTED"

    def submit_jobs(self, tasks: List) -> List[Tuple[bool, str]]:
        """Submit multiple jobs."""
        return [self.submit_job(task) for task in tasks]


def _make_slurm_default(
    default_cpus=4,
    default_memory=32,
    gpu_count=1,
    default_time="05:00:00",
):
    default_partition = "htc-el8"
    if gpu_count > 0:
        default_partition = "gpu-el8"

    args = {
        "job-name": "{job_name}",
        "output": "{output_file}",
        "error": "{error_file}",
        "ntasks": "1",
        "nodes": "1",
        "ntasks-per-node": "1",
        "cpus-per-task": str(default_cpus),
        "mem": f"{default_memory}G",
        "time": default_time,
        "partition": default_partition,
        "qos": "normal",
        "export": "none",
    }

    if gpu_count > 0:
        args["gres"] = f"gpu:{gpu_count}"
    return " ".join([f"--{param}={value}" for param, value in args.items()])


def add_job_submission(parser, output_dir="./results", submit_args=""):
    """Add job submission arguments to a parser.

    Returns
    -------
    Set[str]
        Set of argument destination names added by this function.
    """
    job_group = parser.add_argument_group("Job Submission")
    job_group.add_argument(
        "--output-dir",
        type=Path,
        default=output_dir,
        help="Output directory for results",
    )
    job_group.add_argument(
        "--script-dir",
        type=Path,
        required=False,
        help="Directory for generated scripts",
    )
    job_group.add_argument(
        "--environment-setup",
        default="module load pyTME",
        help="Command(s) to set up environment",
    )
    job_group.add_argument(
        "--submit-command",
        default="sbatch",
        help="Command to submit scripts to queue system "
        "(sbatch (SLURM), qsub (PBS/SGE), bsub (LSF), bash (local))",
    )
    job_group.add_argument(
        "--submit-args",
        default=submit_args,
        help="Arguments passed to submit command. Use {job_name}, {output_file}, "
        "{error_file} for variable substitution.",
    )
    job_group.add_argument(
        "--dry-run", action="store_true", help="Generate scripts but do not submit jobs"
    )
    job_group.add_argument("--force", action="store_true", help="Rerun completed jobs")


def add_discovery_options(group):
    """Add discovery verbosity options to a parser.

    Returns
    -------
    Set[str]
        Set of argument destination names added by this function.
    """
    group.add_argument(
        "--quiet",
        help=argparse.SUPPRESS,
        action=cli.DeprecatedAction,
    )
    group.add_argument(
        "--strict",
        action="store_true",
        help="Raise error if any provided pattern doesn't match all inputs",
    )


def add_matching_discovery(parser):
    """Add dataset discovery arguments for template matching.

    Returns
    -------
    Set[str]
        Set of argument destination names added by this function.
    """
    discovery_group = parser.add_argument_group("Dataset Discovery")
    discovery_group.add_argument(
        "--tomograms",
        required=True,
        help="Glob pattern for tomogram files (e.g., '/data/tomograms/*.mrc')",
    )
    discovery_group.add_argument(
        "--metadata",
        required=False,
        help="Glob pattern for metadata files (e.g., '/data/metadata/*.xml')",
    )
    discovery_group.add_argument(
        "--masks",
        help="Glob pattern for target mask files (e.g., '/data/masks/*.mrc')",
    )
    discovery_group.add_argument(
        "--orientations",
        required=False,
        help="Glob pattern for seed point files (e.g., '/data/seed_points/*.star')",
    )
    discovery_group.add_argument(
        "--tomo-list",
        type=Path,
        help="File with list of tomogram IDs to process (one per line)",
    )
    discovery_group.add_argument(
        "--output-format",
        choices=["pickle", "pickle.gz", "hdf5"],
        help="Output format. pickle.gz and hdf5 yield much smaller files for "
        "constrained matching.",
    )
    add_discovery_options(discovery_group)


def add_analysis_discovery(parser):
    """Add dataset discovery arguments for analysis.

    Returns
    -------
    Set[str]
        Set of argument destination names added by this function.
    """
    discovery_group = parser.add_argument_group("Dataset Discovery")
    discovery_group.add_argument(
        "--input-files",
        required=True,
        nargs="+",
        help="Glob patterns for TM result files (e.g., '/data/results/*.pickle')",
    )
    discovery_group.add_argument(
        "--background-files",
        required=False,
        nargs="+",
        default=[],
        help="Glob patterns for background/normalization files.",
    )
    discovery_group.add_argument(
        "--masks",
        help="Glob pattern for target mask files (e.g., '/data/masks/*.mrc')",
    )
    discovery_group.add_argument(
        "--tomo-list",
        type=Path,
        help="File with list of tomogram IDs to process (one per line)",
    )
    add_discovery_options(discovery_group)


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
    add_matching_discovery(matching_parser)
    add_job_submission(
        matching_parser,
        output_dir="./matching_results",
        submit_args=_make_slurm_default(
            default_cpus=4, default_memory=32, gpu_count=1, default_time="05:00:00"
        ),
    )
    matching_args = add_matching_arguments(matching_parser, batch_mode=True)

    analysis_parser = subparsers.add_parser(
        "analysis",
        help="Analyze template matching results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_analysis_discovery(analysis_parser)
    add_job_submission(
        analysis_parser,
        output_dir="./analysis_results",
        submit_args=_make_slurm_default(
            default_cpus=2, default_memory=16, gpu_count=0, default_time="01:00:00"
        ),
    )
    analysis_args = add_postprocess_arguments(analysis_parser, batch_mode=True)

    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Optimize template matching parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    optimize_parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Manifest YAML from pytme optimize --dry-run.",
    )
    optimize_parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Output prefix passed to each pytme optimize call.",
    )
    optimize_parser.add_argument(
        "--runs-per-job",
        type=int,
        default=1,
        help="Number of matching configs per submitted job.",
    )
    optimize_parser.add_argument(
        "--no-match",
        action="store_true",
        help="Skip matching in all jobs.",
    )
    optimize_parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Skip postprocessing in all jobs.",
    )
    add_job_submission(
        optimize_parser,
        output_dir="./optimize_results",
        submit_args=_make_slurm_default(
            default_cpus=4, default_memory=32, gpu_count=1, default_time="05:00:00"
        ),
    )

    args = parser.parse_args()
    if args.command != "optimize":
        if args.tomo_list is not None:
            with open(args.tomo_list, mode="r") as f:
                args.tomo_list = [line.strip() for line in f if line.strip()]

    args.output_dir = args.output_dir.absolute()

    return args, matching_args, analysis_args


def setup_matching(args, cli_args: cli.ArgumentMetadata):
    cli.print_block(
        name="Discovering Dataset",
        data={
            "Tomogram Pattern": args.tomograms,
            "Metadata Pattern": args.metadata,
            "Mask Pattern": args.masks,
            "Orientation Pattern": args.orientations,
        },
        label_width=30,
        logger=logger,
    )
    logger.info("\n" + "-" * 80 + "\n")

    discovery = TomoDatasetDiscovery(
        mrc_pattern=args.tomograms,
        metadata_pattern=args.metadata,
        mask_pattern=args.masks,
        orientation_pattern=args.orientations,
        strict=args.strict,
    )
    files = discovery.discover(tomo_list=args.tomo_list)

    return [
        TemplateMatchingTask(
            files=file, args=args, output_dir=args.output_dir, cli_args=cli_args
        )
        for file in files
    ]


def setup_analysis(args, cli_args: cli.ArgumentMetadata):
    cli.print_block(
        name="Discovering Dataset",
        data={
            "Input Patterns": args.input_files,
            "Background Patterns": args.background_files,
            "Mask Pattern": args.masks,
        },
        label_width=30,
        logger=logger,
    )
    logger.info("\n" + "-" * 80 + "\n")

    discovery = AnalysisDatasetDiscovery(
        input_patterns=args.input_files,
        background_patterns=args.background_files,
        mask_patterns=args.masks,
        strict=args.strict,
    )
    files = discovery.discover(tomo_list=args.tomo_list)

    return [
        AnalysisTask(
            files=file, args=args, output_dir=args.output_dir, cli_args=cli_args
        )
        for file in files
    ]


def setup_optimize(args):
    """Create optimize tasks by chunking manifest indices."""
    import yaml

    with open(args.manifest) as f:
        manifest = yaml.safe_load(f)

    from tme.scripts.optimize import _build_grid

    matching_grid = _build_grid(manifest.get("matching", {}))
    n_configs = len(matching_grid)
    runs_per_job = args.runs_per_job

    logger.info("Manifest: %s (%d matching configs)", args.manifest, n_configs)
    logger.info(
        "Runs per job: %d -> %d jobs", runs_per_job, -(-n_configs // runs_per_job)
    )

    tasks = []
    for start in range(0, n_configs, runs_per_job):
        indices = list(range(start, min(start + runs_per_job, n_configs)))
        tag = str(start) if len(indices) == 1 else f"{start}_{indices[-1]}"
        files = OptimizeFiles(
            tomo_id=f"opt_{tag}",
            manifest=args.manifest,
            output_prefix=args.output_prefix,
            indices=indices,
        )
        tasks.append(
            OptimizeTask(
                files=files,
                args=args,
                output_dir=args.output_dir,
            )
        )

    return tasks


def main():
    logging.setup_logging()
    cli.print_entry(logger)

    args, matching_args, analysis_args = parse_args()

    if args.command == "optimize":
        tasks = setup_optimize(args)
    elif args.command == "analysis":
        tasks = setup_analysis(args, analysis_args)
    else:
        tasks = setup_matching(args, matching_args)

    if not tasks:
        exit("No files were found for task creation. Check your patterns.")

    backend = ExecutionBackend(
        submit_command=args.submit_command,
        submit_args=args.submit_args,
        script_dir=args.script_dir,
        dry_run=args.dry_run,
        force=args.force,
        environment_setup=args.environment_setup,
    )
    job_ids = backend.submit_jobs(tasks)
    if args.dry_run:
        job_ids = [x for submitted, x in job_ids if submitted]
        exit(f"Dry run done. {len(job_ids)} tasks are not yet completed.")

    successful_jobs = [j for k, j in job_ids if k]
    skipped_jobs = [
        (task.tomo_id, msg) for task, (ok, msg) in zip(tasks, job_ids) if not ok
    ]

    logger.info(f"Submitted {len(successful_jobs)} jobs successfully.")
    if successful_jobs:
        logger.info(f"Job IDs: {','.join(successful_jobs).strip()}")

    if skipped_jobs:
        logger.warning(f"Skipped {len(skipped_jobs)} jobs:")
        for tomo_id, reason in skipped_jobs:
            logger.warning(f"  {tomo_id}: {reason}")


if __name__ == "__main__":
    main()
