""" Combine filters using an interface analogous to pytorch's Compose.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, Dict

from tme.backends import backend as be


class Compose:
    """
    Compose a series of transformations.

    This class allows composing multiple transformations together. Each transformation
    is expected to be a callable that accepts keyword arguments and returns metadata.

    Parameters:
    -----------
    transforms : Tuple[object]
        A tuple containing transformation objects.

    Returns:
    --------
    Dict
        Metadata resulting from the composed transformations.

    """

    def __init__(self, transforms: Tuple[object]):
        self.transforms = transforms

    def __call__(self, **kwargs: Dict) -> Dict:
        meta = {}
        if not len(self.transforms):
            return meta

        meta = self.transforms[0](**kwargs)
        for transform in self.transforms[1:]:
            kwargs.update(meta)
            ret = transform(**kwargs)

            if "data" not in ret:
                continue

            if ret.get("is_multiplicative_filter", False):
                prev_data = meta.pop("data")
                ret["data"] = be.multiply(ret["data"], prev_data, out=ret["data"])
                ret["merge"], prev_data = None, None

            meta = ret

        return meta
