import pytest

from tme.filters import Compose, ComposableFilter
from tme.backends import backend as be


class MockFilter(ComposableFilter):
    def _evaluate(self, *args, **kwargs):
        return {"data": be.ones((10, 10)), "shape": (10, 10)}


class MockFilterNoMult(ComposableFilter):
    def _evaluate(self, *args, **kwargs):
        return {
            "data": be.ones((10, 10)) * 3,
            "shape": (10, 10),
            "is_multiplicative_filter": False,
        }


mock_transform = MockFilter()
mock_transform_nomult = MockFilterNoMult()


def mock_transform_error(**kwargs):
    return {"data": be.ones((10, 10)), "is_multiplicative_filter": True}


class TestCompose:
    @pytest.fixture
    def compose_instance(self):
        return Compose((mock_transform, mock_transform, mock_transform))

    def test_init(self):
        transforms = (mock_transform, mock_transform)
        compose = Compose(transforms)
        assert compose.transforms == transforms

    def test_call_empty_transforms(self):
        compose = Compose(())
        result = compose()
        assert result == {}

    def test_call_single_transform(self):
        compose = Compose((mock_transform,))
        result = compose(return_real_fourier=False)
        assert "data" in result
        assert be.allclose(result["data"], be.ones((10, 10)))

    def test_call_multiple_transforms(self, compose_instance):
        result = compose_instance(return_real_fourier=False)
        assert "data" in result
        assert be.allclose(result["data"], be.ones((10, 10)))

    def test_multiplicative_filter_composition(self):
        compose = Compose((mock_transform, mock_transform))
        result = compose(return_real_fourier=False)
        assert "data" in result
        assert be.allclose(result["data"], be.ones((10, 10)))

    @pytest.mark.parametrize(
        "kwargs", [{}, {"extra_param": "test"}, {"data": be.zeros((5, 5))}]
    )
    def test_call_with_kwargs(self, compose_instance, kwargs):
        result = compose_instance(**kwargs, return_real_fourier=False)
        assert "data" in result
        assert "extra_info" not in result

    def test_non_multiplicative_filter(self):
        compose = Compose((mock_transform, mock_transform_nomult))
        result = compose(return_real_fourier=False)
        assert "data" in result
        assert be.allclose(result["data"], be.ones((10, 10)) * 3)

    def test_error_handling(self):
        with pytest.raises(ValueError):
            Compose((mock_transform, mock_transform_error))()
