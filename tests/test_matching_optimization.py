import numpy as np
import pytest

from tme.matching_utils import euler_from_rotationmatrix
from tme.matching_optimization import (
    MATCHING_OPTIMIZATION_REGISTER,
    register_matching_optimization,
    _MatchCoordinatesToDensity,
    _MatchCoordinatesToCoordinates,
    optimize_match,
    create_score_object,
)

density_to_density = ["FLC"]

coordinate_to_density = [
    k
    for k, v in MATCHING_OPTIMIZATION_REGISTER.items()
    if issubclass(v, _MatchCoordinatesToDensity)
]

coordinate_to_coordinate = [
    k
    for k, v in MATCHING_OPTIMIZATION_REGISTER.items()
    if issubclass(v, _MatchCoordinatesToCoordinates)
]


class TestMatchDensityToDensity:
    def setup_method(self):
        target = np.zeros((50, 50, 50))
        target[20:30, 30:40, 12:17] = 1
        self.target = target
        self.template = target.copy()
        self.template_mask = np.ones_like(target)

    def teardown_method(self):
        self.target = None
        self.template = None
        self.template_mask = None

    @pytest.mark.parametrize("method", density_to_density)
    def test_initialization(self, method: str, notest: bool = False):
        instance = create_score_object(
            score=method,
            target=self.target,
            template=self.template,
            template_mask=self.template_mask,
        )
        if notest:
            return instance

    @pytest.mark.parametrize("method", density_to_density)
    def test_call(self, method):
        instance = self.test_initialization(method=method, notest=True)
        score = instance()
        assert isinstance(score, float)


class TestMatchDensityToCoordinates:
    def setup_method(self):
        data = np.zeros((50, 50, 50))
        data[20:30, 30:40, 12:17] = 1
        self.target = data
        self.target_mask_density = data > 0
        self.coordinates = np.array(np.where(self.target > 0))
        self.coordinates_weights = self.target[tuple(self.coordinates)]

        np.random.seed(42)
        random_pixels = np.random.choice(
            range(self.coordinates.shape[1]), self.coordinates.shape[1] // 2
        )
        self.coordinates_mask = self.coordinates[:, random_pixels]

        self.origin = np.zeros(self.coordinates.shape[0])
        self.sampling_rate = np.ones(self.coordinates.shape[0])

    def teardown_method(self):
        self.target = None
        self.target_mask_density = None
        self.coordinates = None
        self.coordinates_weights = None
        self.coordinates_mask = None

    @pytest.mark.parametrize("method", coordinate_to_density)
    def test_initialization(self, method: str, notest: bool = False):
        instance = create_score_object(
            score=method,
            target=self.target,
            target_mask=self.target_mask_density,
            template_coordinates=self.coordinates,
            template_weights=self.coordinates_weights,
            template_mask_coordinates=self.coordinates_mask,
        )
        if notest:
            return instance

    @pytest.mark.parametrize("method", coordinate_to_density)
    def test_call(self, method):
        instance = self.test_initialization(method=method, notest=True)
        score = instance()
        assert isinstance(score, float)


class TestMatchCoordinateToCoordinates:
    def setup_method(self):
        data = np.zeros((50, 50, 50))
        data[20:30, 30:40, 12:17] = 1
        self.target_coordinates = np.array(np.where(data > 0))
        self.target_weights = data[tuple(self.target_coordinates)]

        self.coordinates = np.array(np.where(data > 0))
        self.coordinates_weights = data[tuple(self.coordinates)]

        self.origin = np.zeros(self.coordinates.shape[0])
        self.sampling_rate = np.ones(self.coordinates.shape[0])

    def teardown_method(self):
        self.target_coordinates = None
        self.target_weights = None
        self.coordinates = None
        self.coordinates_weights = None

    @pytest.mark.parametrize("method", coordinate_to_coordinate)
    def test_initialization(self, method: str, notest: bool = False):
        instance = create_score_object(
            score=method,
            target_coordinates=self.target_coordinates,
            target_weights=self.target_weights,
            template_coordinates=self.coordinates,
            template_weights=self.coordinates_weights,
        )
        if notest:
            return instance

    @pytest.mark.parametrize("method", coordinate_to_coordinate)
    def test_call(self, method):
        instance = self.test_initialization(method=method, notest=True)
        score = instance()
        assert isinstance(score, float)


class TestOptimizeMatch:
    def setup_method(self):
        data = np.zeros((50, 50, 50))
        data[20:30, 30:40, 12:17] = 1
        self.target = data
        self.coordinates = np.array(np.where(self.target > 0))
        self.coordinates_weights = self.target[tuple(self.coordinates)]

        self.origin = np.zeros(self.coordinates.shape[0])
        self.sampling_rate = np.ones(self.coordinates.shape[0])

        self.score_object = MATCHING_OPTIMIZATION_REGISTER["CrossCorrelation"]
        self.score_object = self.score_object(
            target=self.target,
            template_coordinates=self.coordinates,
            template_weights=self.coordinates_weights,
        )

    def teardown_method(self):
        self.target = None
        self.coordinates = None
        self.coordinates_weights = None

    @pytest.mark.parametrize(
        "method", ("differential_evolution", "basinhopping", "minimize")
    )
    @pytest.mark.parametrize("bound_translation", (True, False))
    @pytest.mark.parametrize("bound_rotation", (True, False))
    def test_call(self, method, bound_translation, bound_rotation):
        if bound_rotation:
            bound_rotation = tuple((-90, 90) for _ in range(self.target.ndim))
        else:
            bound_rotation = None

        if bound_translation:
            bound_translation = tuple((-5, 5) for _ in range(self.target.ndim))
        else:
            bound_translation = None

        translation, rotation, score = optimize_match(
            score_object=self.score_object,
            optimization_method=method,
            bounds_rotation=bound_rotation,
            bounds_translation=bound_translation,
            maxiter=10,
        )
        assert translation.size == self.target.ndim
        assert rotation.shape[0] == self.target.ndim
        assert rotation.shape[1] == self.target.ndim
        assert isinstance(score, float)

        if bound_translation is not None:
            lower_bound = np.array([x[0] for x in bound_translation])
            upper_bound = np.array([x[1] for x in bound_translation])
            assert np.all(
                np.logical_and(translation >= lower_bound, translation <= upper_bound)
            )

        if bound_rotation is not None:
            angles = euler_from_rotationmatrix(rotation)
            lower_bound = np.array([x[0] for x in bound_rotation])
            upper_bound = np.array([x[1] for x in bound_rotation])
            assert np.all(np.logical_and(angles >= lower_bound, angles <= upper_bound))

    def test_call_error(self):
        with pytest.raises(ValueError):
            translation, rotation, score = optimize_match(
                score_object=self.score_object,
                optimization_method="RAISERROR",
                maxiter=10,
            )


class TestUtils:
    def test_register_matching_optimization(self):
        new_class = list(MATCHING_OPTIMIZATION_REGISTER.keys())[0]
        register_matching_optimization(
            match_name="new_score",
            match_class=MATCHING_OPTIMIZATION_REGISTER[new_class],
        )

    def test_register_matching_optimization_error(self):
        with pytest.raises(ValueError):
            register_matching_optimization(match_name="new_score", match_class=None)
