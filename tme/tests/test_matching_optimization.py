import numpy as np
import pytest

from tme.matching_optimization import (
    FitRefinement,
    MATCHING_OPTIMIZATION_REGISTER,
    register_matching_optimization,
)


class TestMatchOptimization:
    def setup_method(self):
        data = np.zeros((50, 50, 50))
        data[20:30, 30:40, 12:17] = 1
        self.data = data
        self.coordinates = np.array(np.where(data > 0))
        self.coordinates_weights = self.data[tuple(self.coordinates)]

        self.origin = np.zeros(self.coordinates.shape[0])
        self.sampling_rate = np.ones(self.coordinates.shape[0])

    def teardown_method(self):
        self.data = None
        self.coordinates = None
        self.coordinates_weights = None

    def test_initialization(self):
        _ = FitRefinement()

    def test_map_coordinates_to_array(self):
        ret = FitRefinement.map_coordinates_to_array(
            coordinates=self.coordinates,
            array_shape=self.data.shape,
            array_origin=np.zeros(self.data.ndim),
            sampling_rate=np.ones(self.data.ndim),
        )
        assert len(ret) == 4

        coord, coord_mask, in_vol, in_vol_mask = ret

        assert coord_mask is None
        assert in_vol_mask is None

        assert np.allclose(coord.shape, self.coordinates.shape)
        assert np.allclose(in_vol.shape, self.coordinates.shape[1])

    def test_map_coordinates_to_array_mask(self):
        ret = FitRefinement.map_coordinates_to_array(
            coordinates=self.coordinates,
            array_shape=self.data.shape,
            array_origin=self.origin,
            sampling_rate=self.sampling_rate,
            coordinates_mask=self.coordinates,
        )
        assert len(ret) == 4

        coord, coord_mask, in_vol, in_vol_mask = ret

        assert np.allclose(coord, coord_mask)
        assert np.allclose(in_vol, in_vol_mask)

    def test_array_from_coordinates(self):
        ret = FitRefinement.array_from_coordinates(
            coordinates=self.coordinates,
            weights=self.coordinates_weights,
            sampling_rate=self.sampling_rate,
        )
        assert len(ret) == 3
        arr, positions, origin = ret
        assert arr.ndim == self.coordinates.shape[0]
        assert positions.shape == self.coordinates.shape
        assert origin.shape == (self.coordinates.shape[0],)

        assert np.allclose(origin, self.coordinates.min(axis=1))

        ret = FitRefinement.array_from_coordinates(
            coordinates=self.coordinates,
            weights=self.coordinates_weights,
            sampling_rate=self.sampling_rate,
            origin=self.origin,
        )
        arr, positions, origin = ret
        assert np.allclose(origin, self.origin)

    @pytest.mark.parametrize(
        "scoring_class", list(MATCHING_OPTIMIZATION_REGISTER.keys())
    )
    @pytest.mark.parametrize("local_optimization", (False, True))
    def test_refine_base(self, scoring_class, local_optimization: bool):
        class_object = FitRefinement()
        target_coordinates = np.array(np.where(self.data > 0))
        class_object.refine(
            target_coordinates=target_coordinates,
            target_weights=self.data[tuple(target_coordinates)],
            template_coordinates=self.coordinates,
            template_weights=self.coordinates_weights,
            scoring_class=scoring_class,
            maxiter=1,
            scoring_class_parameters={
                "target_threshold": 0.2,
                "target_mask_coordinates": np.array(np.where(self.data > 0)),
                "template_mask_coordinates": self.coordinates[:, 0:50],
            },
            local_optimization=local_optimization,
        )

    def test_refine_error(self):
        class_object = FitRefinement()

        target_coordinates = np.array(np.where(self.data > 0))
        with pytest.raises(NotImplementedError):
            class_object.refine(
                target_coordinates=target_coordinates,
                target_weights=self.data[tuple(target_coordinates)],
                template_coordinates=self.coordinates,
                template_weights=self.coordinates_weights,
                scoring_class=None,
                maxiter=1,
            )

    def test_register_matching_optimization(self):
        new_class = list(MATCHING_OPTIMIZATION_REGISTER.keys())[0]
        register_matching_optimization(
            match_name="new_score",
            match_class=MATCHING_OPTIMIZATION_REGISTER[new_class],
        )

    def test_register_matching_optimization_error(self):
        with pytest.raises(ValueError):
            register_matching_optimization(match_name="new_score", match_class=None)
