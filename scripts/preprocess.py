#!python3
""" Apply tme.preprocessor.Preprocessor methods to an input file based
    on a provided yaml configuration obtaiend from preprocessor_gui.py.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import yaml
import argparse
import textwrap
from tme import Preprocessor, Density


def parse_args():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """
        Apply preprocessing to an input file based on a provided YAML configuration.

        Expected YAML file format:
        ```yaml
        <method_name>:
            <parameter1>: <value1>
            <parameter2>: <value2>
            ...
        ```
        """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Path to the input data file in CCP4/MRC format.",
    )
    parser.add_argument(
        "-y",
        "--yaml_file",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        required=True,
        help="Path to output file in CPP4/MRC format..",
    )
    parser.add_argument(
        "--compress", action="store_true", help="Compress the output file using gzip."
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    with open(args.yaml_file, "r") as f:
        preprocess_settings = yaml.safe_load(f)

    if len(preprocess_settings) > 1:
        raise NotImplementedError(
            "Multiple preprocessing methods specified. "
            "The script currently supports one method at a time."
        )

    method_name = list(preprocess_settings.keys())[0]
    if not hasattr(Preprocessor, method_name):
        raise ValueError(f"Method {method_name} does not exist in Preprocessor.")

    density = Density.from_file(args.input_file)
    output = density.empty

    method_params = preprocess_settings[method_name]
    preprocessor = Preprocessor()
    method = getattr(preprocessor, method_name, None)
    if not method:
        raise ValueError(
            f"{method} does not exist in dge.preprocessor.Preprocessor class."
        )

    output.data = method(template=density.data, **method_params)
    output.to_file(args.output_file, gzip=args.compress)


if __name__ == "__main__":
    main()
