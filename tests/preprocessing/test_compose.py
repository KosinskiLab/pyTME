import pytest

from tme.filters import Compose
from tme.backends import backend as be


def mock_transform1(**kwargs):
    return {"data": be.ones((10, 10)), "is_multiplicative_filter": True}


def mock_transform2(**kwargs):
    return {"data": be.ones((10, 10)) * 2, "is_multiplicative_filter": True}


def mock_transform3(**kwargs):
    return {"extra_info": "test"}


class TestCompose:
    @pytest.fixture
    def compose_instance(self):
        return Compose((mock_transform1, mock_transform2, mock_transform3))

    def test_init(self):
        transforms = (mock_transform1, mock_transform2)
        compose = Compose(transforms)
        assert compose.transforms == transforms

    def test_call_empty_transforms(self):
        compose = Compose(())
        result = compose()
        assert result == {}

    def test_call_single_transform(self):
        compose = Compose((mock_transform1,))
        result = compose()
        assert "data" in result
        assert result.get("is_multiplicative_filter", False)
        assert be.allclose(result["data"], be.ones((10, 10)))

    def test_call_multiple_transforms(self, compose_instance):
        result = compose_instance()
        assert "data" in result
        assert "extra_info" not in result
        assert be.allclose(result["data"], be.ones((10, 10)) * 2)

    def test_multiplicative_filter_composition(self):
        compose = Compose((mock_transform1, mock_transform2))
        result = compose()
        assert "data" in result
        assert be.allclose(result["data"], be.ones((10, 10)) * 2)

    @pytest.mark.parametrize(
        "kwargs", [{}, {"extra_param": "test"}, {"data": be.zeros((5, 5))}]
    )
    def test_call_with_kwargs(self, compose_instance, kwargs):
        result = compose_instance(**kwargs)
        assert "data" in result
        assert "extra_info" not in result

    def test_non_multiplicative_filter(self):
        def non_mult_transform(**kwargs):
            return {"data": be.ones((10, 10)) * 3, "is_multiplicative_filter": False}

        compose = Compose((mock_transform1, non_mult_transform))
        result = compose()
        assert "data" in result
        assert be.allclose(result["data"], be.ones((10, 10)) * 3)

    def test_error_handling(self):
        def error_transform(**kwargs):
            raise ValueError("Test error")

        compose = Compose((mock_transform1, error_transform))
        with pytest.raises(ValueError, match="Test error"):
            compose()
