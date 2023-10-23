import pytest
from tme import Density
from tme.matching_memory import MATCHING_MEMORY_REGISTRY, estimate_ram_usage


class TestMatchingMemory:
    def setup_method(self):
        self.density = Density.from_file(filename="./tme/tests/data/Raw/em_map.map")
        self.structure_density = Density.from_structure(
            filename_or_structure="./tme/tests/data/Structures/5khe.cif",
            origin=self.density.origin,
            shape=self.density.shape,
            sampling_rate=self.density.sampling_rate,
        )

    @pytest.mark.parametrize("analyzer_method", ["MaxScoreOverRotations", None])
    @pytest.mark.parametrize("matching_method", list(MATCHING_MEMORY_REGISTRY.keys()))
    @pytest.mark.parametrize("ncores", range(1, 10, 3))
    def test_estimate_ram_usage(self, matching_method, ncores, analyzer_method):
        estimate_ram_usage(
            shape1=self.density.shape,
            shape2=self.structure_density.shape,
            matching_method=matching_method,
            ncores=ncores,
            analyzer_method=analyzer_method,
        )
