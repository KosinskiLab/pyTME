#!python3

from os import listdir
from os.path import dirname, join, basename

import numpy as np
import yaml


def quat_to_numpy(filepath):
    data = []
    with open(join(filepath, filepath), "r", encoding="utf-8") as ifile:
        for line in ifile:
            if line.startswith("#") or line.startswith("format quaternion"):
                continue
            line_split = line.strip().split()
            if len(line_split) == 3:
                n, angle, c = line_split
                continue
            data.append(line_split)

    data = np.array(data).astype(float)
    return data, int(n), float(angle), float(c)


if __name__ == "__main__":
    current_directory = dirname(__file__)
    files = listdir(current_directory)

    files = [file for file in files if file.endswith(".quat")]
    files = [join(current_directory, file) for file in files]
    numpy_names = [
        join(current_directory, file.replace("quat", "npy")) for file in files
    ]

    metadata = {}
    for file, np_out in zip(files, numpy_names):
        quaternions, n, angle, c = quat_to_numpy(file)
        np.save(np_out, quaternions)
        metadata[basename(np_out)] = [n, angle, c]
    with open(join(current_directory, "metadata.yaml"), "w", encoding="utf-8") as ofile:
        yaml.dump(metadata, ofile, default_flow_style=False)
