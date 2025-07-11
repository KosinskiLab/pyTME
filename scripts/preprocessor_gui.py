#!python3
"""
GUI for identifying suitable masks and analyzing template matchign results.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import inspect
import argparse
from os.path import basename
from typing_extensions import Annotated
from typing import Tuple, Callable, List

import napari
import numpy as np
import pandas as pd
from magicgui import widgets
from numpy.typing import NDArray
from napari.layers import Image
from scipy.fft import next_fast_len
from napari.utils.events import EventedList
from qtpy.QtWidgets import QFileDialog, QMessageBox

from tme.backends import backend
from tme.rotations import align_vectors
from tme import Preprocessor, Density, Orientations
from tme.matching_utils import create_mask, load_pickle
from tme.filters import BandPassReconstructed, CTFReconstructed

preprocessor = Preprocessor()
SLIDER_MIN, SLIDER_MAX = 0, 25


def gaussian_filter(template: NDArray, sigma: float, **kwargs: dict) -> NDArray:
    return preprocessor.gaussian_filter(template=template, sigma=sigma, **kwargs)


def bandpass_filter(
    template: NDArray,
    lowpass_angstrom: float = 30,
    highpass_angstrom: float = 140,
    hard_edges: bool = False,
    sampling_rate=None,
) -> NDArray:
    bpf = BandPassReconstructed(
        lowpass=lowpass_angstrom,
        highpass=highpass_angstrom,
        sampling_rate=np.max(sampling_rate),
        use_gaussian=not hard_edges,
        return_real_fourier=True,
    )
    template_ft = np.fft.rfftn(template, s=template.shape)

    mask = bpf(shape=template.shape)["data"]
    np.multiply(template_ft, mask, out=template_ft)
    return np.fft.irfftn(template_ft, s=template.shape).real


def ctf_filter(
    template: NDArray,
    defocus_angstrom: float = 30000,
    acceleration_voltage: float = 300,
    spherical_aberration: float = 2.7,
    amplitude_contrast: float = 0.07,
    phase_shift: float = 0,
    defocus_angle: float = 0,
    sampling_rate=None,
    flip_phase: bool = False,
) -> NDArray:
    fast_shape = [next_fast_len(x) for x in np.multiply(template.shape, 2)]
    template_pad = backend.topleft_pad(template, fast_shape)
    template_ft = np.fft.rfftn(template_pad, s=template_pad.shape)
    ctf = CTFReconstructed(
        shape=fast_shape,
        defocus_x=[defocus_angstrom],
        acceleration_voltage=acceleration_voltage * 1e3,
        spherical_aberration=spherical_aberration * 1e7,
        amplitude_contrast=amplitude_contrast,
        phase_shift=phase_shift,
        defocus_angle=defocus_angle,
        sampling_rate=np.max(sampling_rate),
        return_real_fourier=True,
        flip_phase=flip_phase,
    )
    np.multiply(template_ft, ctf()["data"], out=template_ft)
    template_pad = np.fft.irfftn(template_ft, s=template_pad.shape).real
    template = backend.topleft_pad(template_pad, template.shape)
    return template


def difference_of_gaussian_filter(
    template: NDArray, sigmas: Tuple[float, float], **kwargs: dict
) -> NDArray:
    low_sigma, high_sigma = sigmas
    return preprocessor.difference_of_gaussian_filter(
        template=template, low_sigma=low_sigma, high_sigma=high_sigma, **kwargs
    )


def edge_gaussian_filter(
    template: NDArray,
    sigma: float,
    edge_algorithm: Annotated[
        str,
        {"choices": ["sobel", "prewitt", "laplace", "gaussian", "gaussian_laplace"]},
    ],
    reverse: bool = False,
    **kwargs: dict,
) -> NDArray:
    return preprocessor.edge_gaussian_filter(
        template=template,
        sigma=sigma,
        reverse=reverse,
        edge_algorithm=edge_algorithm,
    )


def local_gaussian_filter(
    template: NDArray,
    lbd: float,
    sigma_range: Tuple[float, float],
    gaussian_sigma: float,
    reverse: bool = False,
    **kwargs: dict,
) -> NDArray:
    return preprocessor.local_gaussian_filter(
        template=template,
        lbd=lbd,
        sigma_range=sigma_range,
        gaussian_sigma=gaussian_sigma,
    )


def mean(
    template: NDArray,
    width: int,
    **kwargs: dict,
) -> NDArray:
    return preprocessor.mean_filter(template=template, width=width)


def wedge(
    template: NDArray,
    tilt_start: float = 40.0,
    tilt_stop: float = 40.0,
    tilt_step: float = 0,
    opening_axis: int = 2,
    tilt_axis: int = 0,
    omit_negative_frequencies: bool = False,
    infinite_plane: bool = False,
    weight_angle: bool = False,
    **kwargs,
) -> NDArray:
    template_ft = np.fft.fftn(template)

    if tilt_step <= 0:
        wedge_mask = preprocessor.continuous_wedge_mask(
            start_tilt=tilt_start,
            stop_tilt=tilt_stop,
            tilt_axis=tilt_axis,
            opening_axis=opening_axis,
            shape=template.shape,
            omit_negative_frequencies=omit_negative_frequencies,
            infinite_plane=infinite_plane,
        )
    else:
        weights = None
        tilt_angles = np.arange(-tilt_start, tilt_stop + tilt_step, tilt_step)
        if weight_angle:
            weights = np.cos(np.radians(tilt_angles))

        wedge_mask = preprocessor.step_wedge_mask(
            tilt_angles=tilt_angles,
            tilt_axis=tilt_axis,
            opening_axis=opening_axis,
            shape=template.shape,
            weights=weights,
            omit_negative_frequencies=omit_negative_frequencies,
        )

    np.multiply(template_ft, wedge_mask, out=template_ft)
    template = np.real(np.fft.ifftn(template_ft))
    return template


def compute_power_spectrum(template: NDArray) -> NDArray:
    return np.fft.fftshift(np.log(np.abs(np.fft.fftn(template))))


def invert_contrast(template: NDArray) -> NDArray:
    return template * -1


def widgets_from_function(function: Callable, exclude_params: List = ["self"]):
    """
    Creates list of magicui widgets by inspecting function typing ann
    """
    ret = []
    for name, param in inspect.signature(function).parameters.items():
        if name in exclude_params:
            continue

        if param.annotation is float:
            widget = widgets.FloatSpinBox(
                name=name,
                value=param.default if param.default != inspect._empty else 0,
                min=SLIDER_MIN,
                step=0.5,
            )
        elif param.annotation == Tuple[float, float]:
            widget = widgets.FloatRangeSlider(
                name=param.name,
                value=(
                    param.default
                    if param.default != inspect._empty
                    else (0.0, SLIDER_MAX / 2)
                ),
                min=SLIDER_MIN,
                max=SLIDER_MAX,
            )
        elif param.annotation is int:
            widget = widgets.SpinBox(
                name=name,
                value=param.default if param.default != inspect._empty else 0,
            )
        elif param.annotation is bool:
            widget = widgets.CheckBox(
                name=name,
                value=param.default if param.default != inspect._empty else False,
            )
        elif hasattr(param.annotation, "__metadata__"):
            metadata = param.annotation.__metadata__[0]
            if "choices" in metadata:
                widget = widgets.ComboBox(
                    name=param.name,
                    choices=metadata["choices"],
                    value=(
                        param.default
                        if param.default != inspect._empty
                        else metadata["choices"][0]
                    ),
                )
        else:
            continue
        ret.append(widget)
    return ret


WRAPPED_FUNCTIONS = {
    "gaussian_filter": gaussian_filter,
    "bandpass_filter": bandpass_filter,
    "edge_gaussian_filter": edge_gaussian_filter,
    "local_gaussian_filter": local_gaussian_filter,
    "difference_of_gaussian_filter": difference_of_gaussian_filter,
    "mean_filter": mean,
    "wedge_filter": wedge,
    "power_spectrum": compute_power_spectrum,
    "ctf": ctf_filter,
    "invert_contrast": invert_contrast,
}

EXCLUDED_FUNCTIONS = [
    "apply_method",
    "method_to_id",
    "wedge_mask",
    "fourier_crop",
    "fourier_uncrop",
    "interpolate_box",
    "molmap",
    "local_gaussian_alignment_filter",
    "continuous_wedge_mask",
    "wedge_mask",
    "bandpass_mask",
]


class FilterWidget(widgets.Container):
    def __init__(self, preprocessor, viewer):
        super().__init__(layout="vertical")

        self.preprocessor = preprocessor
        self.viewer = viewer
        self.name_mapping = {}
        self.action_widgets = []

        self.layer_dropdown = widgets.ComboBox(
            name="Target Layer", choices=self._get_layer_names()
        )
        self.append(self.layer_dropdown)
        self.viewer.layers.events.inserted.connect(self._update_layer_dropdown)
        self.viewer.layers.events.removed.connect(self._update_layer_dropdown)

        self.method_dropdown = widgets.ComboBox(
            name="Choose Filter", choices=self._get_method_names()
        )
        self.method_dropdown.changed.connect(self._on_method_changed)
        self.append(self.method_dropdown)

        self.apply_btn = widgets.PushButton(text="Apply Filter", enabled=False)
        self.apply_btn.changed.connect(self._action)
        self.append(self.apply_btn)

        # Create GUI for initially selected filtering method
        self._on_method_changed(None)

    def _get_method_names(self):
        method_names = [
            name
            for name, member in inspect.getmembers(self.preprocessor, inspect.ismethod)
            if not name.startswith("_") and name not in EXCLUDED_FUNCTIONS
        ]
        method_names.extend(list(WRAPPED_FUNCTIONS.keys()))
        method_names = list(set(method_names))

        sanitized_names = [self._sanitize_name(name) for name in method_names]
        self.name_mapping.update(dict(zip(sanitized_names, method_names)))
        sanitized_names.sort()

        return sanitized_names

    def _sanitize_name(self, name: str) -> str:
        # Replace underscores with spaces and capitalize each word
        removes = ["blur", "filter"]
        for remove in removes:
            name = name.replace(remove, "")
        return name.strip().replace("_", " ").title()

    def _desanitize_name(self, name: str) -> str:
        name = name.lower().strip()
        for function_name, _ in inspect.getmembers(self.preprocessor, inspect.ismethod):
            if function_name.startswith(name):
                return function_name
        return name

    def _get_function(self, name: str):
        function = WRAPPED_FUNCTIONS.get(name, None)
        if not function:
            function = getattr(self.preprocessor, name, None)
        return function

    def _on_method_changed(self, event=None):
        # Clear previous parameter widgets
        for widget in self.action_widgets:
            self.remove(widget)
        self.action_widgets.clear()

        function_name = self.name_mapping.get(self.method_dropdown.value)
        function = self._get_function(function_name)

        widgets = widgets_from_function(function, exclude_params=["self", "template"])
        for widget in widgets:
            self.action_widgets.append(widget)
            self.insert(-1, widget)

    def _update_layer_dropdown(self, event: EventedList):
        """Update the dropdown menu when layers change."""
        self.layer_dropdown.choices = self._get_layer_names()
        self.apply_btn.enabled = bool(self.viewer.layers)

    def _get_layer_names(self):
        """Return list of layer names in the viewer."""
        return sorted([layer.name for layer in self.viewer.layers])

    def _action(self, event):
        selected_layer = self.viewer.layers[self.layer_dropdown.value]
        selected_layer_metadata = selected_layer.metadata.copy()
        kwargs = {widget.name: widget.value for widget in self.action_widgets}

        function_name = self.name_mapping.get(self.method_dropdown.value)
        function = self._get_function(function_name)

        if "sampling_rate" in inspect.getfullargspec(function).args:
            kwargs["sampling_rate"] = selected_layer_metadata["sampling_rate"]

        processed_data = function(selected_layer.data, **kwargs)

        new_layer_name = f"{selected_layer.name} ({self.method_dropdown.value})"

        if new_layer_name in self.viewer.layers:
            selected_layer = self.viewer.layers[new_layer_name]

        filter_name = self._desanitize_name(self.method_dropdown.value)
        used_filter = selected_layer.metadata.get("used_filter", False)
        if used_filter == filter_name:
            selected_layer.data = processed_data
        else:
            new_layer = self.viewer.add_image(
                data=processed_data,
                name=new_layer_name,
            )
            metadata = selected_layer_metadata.copy()
            if "filter_parameters" not in metadata:
                metadata["filter_parameters"] = []

            payload = {filter_name: kwargs.copy()}
            if isinstance(metadata["filter_parameters"], dict):
                metadata["filter_parameters"].update(payload)
            else:
                metadata["filter_parameters"].append(payload)

            metadata["used_filter"] = filter_name
            new_layer.metadata = metadata


def sphere_mask(
    template: NDArray,
    center_x: float,
    center_y: float,
    center_z: float,
    radius: float,
    sigma_decay: float = 0,
    **kwargs,
) -> NDArray:
    return create_mask(
        mask_type="ellipse",
        shape=template.shape,
        center=(center_x, center_y, center_z),
        radius=radius,
        sigma_decay=sigma_decay,
    )


def ellipsod_mask(
    template: NDArray,
    center_x: float,
    center_y: float,
    center_z: float,
    radius_x: float,
    radius_y: float,
    radius_z: float,
    sigma_decay: float = 0,
    **kwargs,
) -> NDArray:
    return create_mask(
        mask_type="ellipse",
        shape=template.shape,
        center=(center_x, center_y, center_z),
        radius=(radius_x, radius_y, radius_z),
        sigma_decay=sigma_decay,
    )


def box_mask(
    template: NDArray,
    center_x: float,
    center_y: float,
    center_z: float,
    height_x: int,
    height_y: int,
    height_z: int,
    sigma_decay: float = 0,
    **kwargs,
) -> NDArray:
    return create_mask(
        mask_type="box",
        shape=template.shape,
        center=(center_x, center_y, center_z),
        height=(height_x, height_y, height_z),
        sigma_decay=sigma_decay,
    )


def membrane_mask(
    template: NDArray,
    symmetry_axis: int,
    center_x: float,
    center_y: float,
    center_z: float,
    radius: float,
    thickness: float = 1,
    separation: float = 3,
    sigma_decay: float = 0,
    **kwargs,
) -> NDArray:
    return create_mask(
        mask_type="membrane",
        shape=template.shape,
        radius=radius,
        thickness=thickness,
        separation=separation,
        sigma_decay=sigma_decay,
    )


def tube_mask(
    template: NDArray,
    symmetry_axis: int,
    center_x: float,
    center_y: float,
    center_z: float,
    inner_radius: float,
    outer_radius: float,
    height: int,
    sigma_decay: float = 0,
    **kwargs,
) -> NDArray:
    return create_mask(
        mask_type="tube",
        shape=template.shape,
        symmetry_axis=symmetry_axis,
        base_center=(center_x, center_y, center_z),
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        height=height,
        sigma_decay=sigma_decay,
    )


def wedge_mask(
    template: NDArray,
    tilt_start: float = 40.0,
    tilt_stop: float = 40.0,
    tilt_step: float = 0,
    opening_axis: int = 2,
    tilt_axis: int = 0,
    omit_negative_frequencies: bool = False,
    infinite_plane: bool = False,
    weight_angle: bool = False,
    **kwargs,
) -> NDArray:
    if tilt_step <= 0:
        wedge_mask = preprocessor.continuous_wedge_mask(
            start_tilt=tilt_start,
            stop_tilt=tilt_stop,
            tilt_axis=tilt_axis,
            opening_axis=opening_axis,
            shape=template.shape,
            omit_negative_frequencies=omit_negative_frequencies,
            infinite_plane=infinite_plane,
        )
        wedge_mask = np.fft.fftshift(wedge_mask)
        return wedge_mask

    weights = None
    tilt_angles = np.arange(-tilt_start, tilt_stop + tilt_step, tilt_step)
    if weight_angle:
        weights = np.cos(np.radians(tilt_angles))

    wedge_mask = preprocessor.step_wedge_mask(
        tilt_angles=tilt_angles,
        tilt_axis=tilt_axis,
        opening_axis=opening_axis,
        shape=template.shape,
        weights=weights,
        omit_negative_frequencies=omit_negative_frequencies,
    )

    wedge_mask = np.fft.fftshift(wedge_mask)
    return wedge_mask


def threshold_mask(
    template: NDArray,
    invert: bool = False,
    standard_deviation: float = 5.0,
    sigma: float = 0.0,
    **kwargs,
) -> NDArray:
    template_mean = template.mean()
    template_deviation = standard_deviation * template.std()
    upper = template_mean + template_deviation
    lower = template_mean - template_deviation
    mask = np.logical_or(template <= lower, template >= upper)

    if sigma != 0:
        mask_filter = preprocessor.gaussian_filter(template=mask * 1.0, sigma=sigma)
        mask = np.add(mask, (1 - mask) * mask_filter)
        mask[mask < np.exp(-np.square(sigma))] = 0

    if invert:
        np.invert(mask, out=mask)

    return mask


def lowpass_mask(template: NDArray, sigma: float = 1.0, **kwargs):
    template = template / template.max()
    template = (template > np.exp(-2)) * 128.0
    template = preprocessor.gaussian_filter(template=template, sigma=sigma)
    mask = template > np.exp(-2)

    return mask


def shape_mask(template, shapes_layer, expansion_dim):
    ret = np.zeros_like(template)
    mask_shape = tuple(x for i, x in enumerate(template.shape) if i != expansion_dim)
    masks = shapes_layer.to_masks(mask_shape=mask_shape)
    for index, shape_type in enumerate(shapes_layer.shape_type):
        mask = np.expand_dims(masks[index], axis=expansion_dim)
        mask = np.repeat(
            mask, repeats=template.shape[expansion_dim], axis=expansion_dim
        )
        np.logical_or(ret, mask, out=ret)

    return ret


class MaskWidget(widgets.Container):
    def __init__(self, viewer):
        super().__init__(layout="vertical")

        self.viewer = viewer
        self.action_widgets = []

        self.action_button = widgets.PushButton(text="Create mask", enabled=False)
        self.action_button.changed.connect(self._action)

        self.methods = {
            "Sphere": sphere_mask,
            "Ellipsoid": ellipsod_mask,
            "Tube": tube_mask,
            "Box": box_mask,
            "Membrane": membrane_mask,
            "Wedge": wedge_mask,
            "Threshold": threshold_mask,
            "Lowpass": lowpass_mask,
            "Shape": shape_mask,
        }

        self.method_dropdown = widgets.ComboBox(
            name="Choose Mask", choices=list(self.methods.keys())
        )
        self.method_dropdown.changed.connect(self._on_method_changed)

        self.percentile_range_edit = widgets.FloatSpinBox(
            name="Data Quantile", min=0, max=100, value=0, step=2
        )

        self.adapt_button = widgets.PushButton(text="Adapt to layer", enabled=False)
        self.adapt_button.changed.connect(self._update_initial_values)
        self.viewer.layers.selection.events.active.connect(
            self._update_action_button_state
        )

        self.density_field = widgets.Label()
        # self.density_field.value = f"Positive Density in Mask: {0:.2f}%"

        self.shapes_layer_dropdown = widgets.ComboBox(
            name="shapes_layer", choices=self._get_shape_layers()
        )
        self.viewer.layers.events.inserted.connect(self._update_shape_layer_choices)
        self.viewer.layers.events.removed.connect(self._update_shape_layer_choices)

        self.append(self.method_dropdown)
        self.append(self.adapt_button)
        self.append(self.percentile_range_edit)

        self.append(self.action_button)
        self.append(self.density_field)

        # Create GUI for initially selected filtering method
        self._on_method_changed(None)

    def _update_action_button_state(self, event):
        self.action_button.enabled = bool(self.viewer.layers.selection.active)
        self.adapt_button.enabled = bool(self.viewer.layers.selection.active)

    def _update_initial_values(self, event=None):
        active_layer = self.viewer.layers.selection.active

        data = active_layer.data.copy()
        cutoff = np.quantile(data, self.percentile_range_edit.value / 100)
        cutoff = max(cutoff, np.finfo(np.float32).resolution)
        data[data < cutoff] = 0

        center_of_mass = Density.center_of_mass(np.abs(data), 0)
        coordinates = np.array(np.where(data > 0))
        coordinates_min = coordinates.min(axis=1)
        coordinates_max = coordinates.max(axis=1)
        coordinates_heights = coordinates_max - coordinates_min
        coordinate_radius = np.divide(coordinates_heights, 2)
        center_of_mass = coordinate_radius + coordinates_min

        defaults = dict(zip(["center_x", "center_y", "center_z"], center_of_mass))
        defaults.update(
            dict(zip(["radius_x", "radius_y", "radius_z"], coordinate_radius))
        )
        defaults.update(
            dict(zip(["height_x", "height_y", "height_z"], coordinates_heights))
        )

        defaults["radius"] = np.max(coordinate_radius)
        defaults["inner_radius"] = np.min(coordinate_radius)
        defaults["outer_radius"] = np.max(coordinate_radius)
        defaults["height"] = np.max(coordinates_heights)

        for widget in self.action_widgets:
            if widget.name in defaults:
                widget.value = defaults[widget.name]

    def _on_method_changed(self, event=None):
        for widget in self.action_widgets:
            self.remove(widget)
        self.action_widgets.clear()

        function = self.methods.get(self.method_dropdown.value)
        function_widgets = widgets_from_function(function)
        for widget in function_widgets:
            self.action_widgets.append(widget)
            self.insert(1, widget)

        for name, param in inspect.signature(function).parameters.items():
            if name == "shapes_layer":
                self.action_widgets.append(self.shapes_layer_dropdown)
                self.insert(1, self.shapes_layer_dropdown)

    def _get_shape_layers(self):
        layers = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Shapes)
        ]
        return layers

    def _update_shape_layer_choices(self, event):
        """Update the choices in the shapes layer dropdown."""
        self.shapes_layer_dropdown.choices = self._get_shape_layers()

    def _action(self):
        function = self.methods.get(self.method_dropdown.value)

        selected_layer = self.viewer.layers.selection.active
        kwargs = {widget.name: widget.value for widget in self.action_widgets}

        if "shapes_layer" in kwargs:
            layer_name = kwargs["shapes_layer"]
            if layer_name not in self.viewer.layers:
                return None
            kwargs["shapes_layer"] = self.viewer.layers[layer_name]
            kwargs["expansion_dim"] = self.viewer.dims.order[0]

        processed_data = function(template=selected_layer.data, **kwargs)

        new_layer_name = f"{selected_layer.name} ({self.method_dropdown.value})"

        if new_layer_name in self.viewer.layers:
            selected_layer = self.viewer.layers[new_layer_name]

        processed_data = processed_data.astype(np.float32)
        metadata = selected_layer.metadata
        mask = metadata.get("mask", False)
        if mask == self.method_dropdown.value:
            selected_layer.data = processed_data
        else:
            new_layer = self.viewer.add_image(
                data=processed_data,
                name=new_layer_name,
            )
            metadata = selected_layer.metadata.copy()
            metadata["filter_parameters"] = {self.method_dropdown.value: kwargs.copy()}
            metadata["mask"] = self.method_dropdown.value
            metadata["origin_layer"] = selected_layer.name
            new_layer.metadata = metadata

            if self.method_dropdown.value == "Shape":
                new_layer.metadata = {}

        # origin_layer = metadata["origin_layer"]
        # if origin_layer in self.viewer.layers:
        #     origin_layer = self.viewer.layers[origin_layer]
        #     if np.allclose(origin_layer.data.shape, processed_data.shape):
        #         in_mask = np.sum(np.fmax(origin_layer.data * processed_data, 0))
        #         in_mask /= np.sum(np.fmax(origin_layer.data, 0))
        #         in_mask *= 100
        #         self.density_field.value = f"Positive Density in Mask: {in_mask:.2f}%"


class AlignmentWidget(widgets.Container):
    def __init__(self, viewer):
        super().__init__(layout="vertical")

        self.viewer = viewer

        align_button = widgets.PushButton(text="Align to axis", enabled=True)
        self.align_axis = widgets.ComboBox(
            value=None, nullable=True, choices=self._get_active_layer_dims
        )
        self.viewer.layers.selection.events.changed.connect(self._update_align_axis)

        align_button.changed.connect(self._align_with_axis)
        container = widgets.Container(
            widgets=[align_button, self.align_axis], layout="horizontal"
        )
        self.append(container)

        rot90 = widgets.PushButton(text="Rotate 90", enabled=True)
        rotneg90 = widgets.PushButton(text="Rotate -90", enabled=True)

        rot90.changed.connect(self._rot90)
        rotneg90.changed.connect(self._rotneg90)

        container = widgets.Container(widgets=[rot90, rotneg90], layout="horizontal")
        self.append(container)

    def _rot90(self, swap_axes: bool = False):
        active_layer = self.viewer.layers.selection.active
        if active_layer is None:
            return None
        elif self.viewer.dims.ndisplay != 2:
            return None

        align_axis = self.align_axis.value
        if self.align_axis.value is None:
            align_axis = self.viewer.dims.order[0]

        axes = [
            align_axis,
            *[i for i in range(len(self.viewer.dims.order)) if i != align_axis],
        ][:2]
        axes = axes[::-1] if swap_axes else axes
        active_layer.data = np.rot90(active_layer.data, k=1, axes=axes)

    def _rotneg90(self):
        return self._rot90(swap_axes=True)

    def _get_active_layer_dims(self, *args):
        active_layer = self.viewer.layers.selection.active
        if active_layer is None:
            return ()
        try:
            return [i for i in range(active_layer.data.ndim)]
        except Exception:
            return ()

    def _update_align_axis(self, *args):
        self.align_axis.choices = self._get_active_layer_dims()

    def _align_with_axis(self):
        active_layer = self.viewer.layers.selection.active

        if self.align_axis.value is None:
            return None

        if active_layer.metadata.get("is_aligned", None) == self.align_axis.value:
            return None

        alignment_axis = np.zeros(active_layer.data.ndim)
        alignment_axis[int(self.align_axis.value)] = 1

        coords = np.array(np.where(active_layer.data > 0)).T
        centered_coords = coords - np.mean(coords, axis=0)
        cov_matrix = np.cov(centered_coords, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        principal_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

        rotation_matrix = align_vectors(principal_eigenvector, alignment_axis)
        rotated_data, _ = backend.rigid_transform(
            arr=active_layer.data,
            rotation_matrix=rotation_matrix,
            use_geometric_center=False,
            order=1,
        )
        eps = np.finfo(rotated_data.dtype).eps
        rotated_data[rotated_data < eps] = 0

        active_layer.metadata["is_aligned"] = int(self.align_axis.value)
        active_layer.data = rotated_data


class ExportWidget(widgets.Container):
    def __init__(self, viewer):
        super().__init__(layout="vertical")

        self.viewer = viewer
        self.selected_filename = ""

        horizontal_container = widgets.Container(layout="horizontal")

        self.gzip_output = widgets.CheckBox(name="gzip", value=False, label="gzip")
        self.export_button = widgets.PushButton(name="Export", text="Export")
        self.export_button.clicked.connect(self._get_save_path)

        horizontal_container.append(self.export_button)
        horizontal_container.append(self.gzip_output)

        self.append(horizontal_container)

        self.export_button.enabled = bool(self.viewer.layers.selection.active)
        self.viewer.layers.selection.events.active.connect(
            self._update_export_button_state
        )

    def _get_save_path(self, event):
        path, _ = QFileDialog().getSaveFileName(
            self.native,
            "Save As...",
            "",
            "MRC Files (*.mrc)",
        )
        if path:
            self.selected_filename = path
            self._export_data()

    def _update_export_button_state(self, event):
        """Update the enabled state of the export button based on the active layer."""
        self.export_button.enabled = bool(self.viewer.layers.selection.active)

    def _export_data(self):
        selected_layer = self.viewer.layers.selection.active
        if selected_layer and isinstance(selected_layer, Image):
            selected_layer.metadata["write_gzip"] = self.gzip_output.value
            selected_layer.save(path=self.selected_filename)


class PointCloudWidget(widgets.Container):
    def __init__(self, viewer):
        super().__init__(layout="vertical")

        self.viewer = viewer
        self.dataframes = {}
        self.selected_category = -1

        self.import_button = widgets.PushButton(
            name="Import", text="Import Point Cloud"
        )
        self.import_button.clicked.connect(self._get_load_path)

        self.export_button = widgets.PushButton(
            name="Export", text="Export Point Cloud"
        )
        self.export_button.clicked.connect(self._export_point_cloud)
        self.export_button.enabled = False

        self.score_filter_container = widgets.Container(
            name="Score Filter", layout="vertical"
        )
        self.score_range_slider = widgets.FloatRangeSlider(
            min=0,
            max=100,
            step=0.1,
            value=(0, 100),
            readout=False,
        )
        self.score_range_slider.changed.connect(self._update_point_visibility)
        self.score_filter_container.append(self.score_range_slider)

        self.annotation_container = widgets.Container(name="Label", layout="horizontal")
        self.positive_button = widgets.PushButton(name="Positive", text="Positive")
        self.negative_button = widgets.PushButton(name="Negative", text="Negative")
        self.positive_button.clicked.connect(self._set_positive)
        self.negative_button.clicked.connect(self._set_negative)
        self.annotation_container.append(self.positive_button)
        self.annotation_container.append(self.negative_button)

        self.face_color_select = widgets.ComboBox(
            name="Color", choices=["Label", "Score"], value=None, nullable=True
        )
        self.face_color_select.changed.connect(self._update_face_color_mode)

        self.append(self.import_button)
        self.append(self.export_button)
        self.append(self.score_filter_container)
        self.append(self.annotation_container)
        self.append(self.face_color_select)

        self.viewer.layers.selection.events.changed.connect(self._update_buttons)

        self.viewer.layers.events.inserted.connect(self._initialize_points_layer)

    def _update_point_visibility(self, event=None):
        min_percentile, max_percentile = self.score_range_slider.value

        for layer in self.viewer.layers:
            if not isinstance(layer, napari.layers.Points):
                continue

            if "score" not in layer.properties:
                continue

            scores = layer.properties["score"]
            layer.selected_data = set()

            min_score = np.percentile(scores, min_percentile)
            max_score = np.percentile(scores, max_percentile)
            visible_mask = (scores >= min_score) & (scores <= max_score)
            layer.shown = visible_mask

    def _update_face_color_mode(self, event: str = None):
        for layer in self.viewer.layers:
            if not isinstance(layer, napari.layers.Points):
                continue

            layer.face_color = "white"
            if event == "Label":
                if len(layer.properties.get("detail", ())) == 0:
                    continue
                layer.face_color = "detail"
                layer.face_color_cycle = {
                    -1: "grey",
                    0: "red",
                    1: "green",
                }
                layer.face_color_mode = "cycle"
            elif event == "Score":
                if len(layer.properties.get("score_scaled", ())) == 0:
                    continue
                layer.face_color = "score_scaled"
                layer.face_colormap = "turbo"
                layer.face_color_mode = "colormap"

            layer.refresh_colors()

        return None

    def _set_positive(self, event):
        self.selected_category = 1 if self.selected_category != 1 else -1
        self._update_annotation_buttons()

    def _set_negative(self, event):
        self.selected_category = 0 if self.selected_category != 0 else -1
        self._update_annotation_buttons()

    def _update_annotation_buttons(self):
        selected_style = "background-color: darkgrey"
        default_style = "background-color: none"

        self.positive_button.native.setStyleSheet(
            selected_style if self.selected_category == 1 else default_style
        )
        self.negative_button.native.setStyleSheet(
            selected_style if self.selected_category == 0 else default_style
        )

    def _initialize_points_layer(self, event):
        layer = event.value
        if not isinstance(layer, napari.layers.Points):
            return
        if len(layer.properties) == 0:
            layer.properties = {"detail": [-1]}

        if "detail" not in layer.properties:
            layer["detail"] = [-1]

        layer.mouse_drag_callbacks.append(self._on_click)
        return None

    def _on_click(self, layer, event):
        if layer.mode == "add":
            layer.current_properties["detail"][-1] = self.selected_category
        elif layer.mode == "select":
            for index in layer.selected_data:
                layer.properties["detail"][index] = self.selected_category

        # TODO: Check whether current face color is the desired one already
        self._update_face_color_mode(self.face_color_select.value)
        layer.refresh_colors()

    def _update_buttons(self, event):
        is_pointcloud = isinstance(
            self.viewer.layers.selection.active, napari.layers.Points
        )
        if self.viewer.layers.selection.active and is_pointcloud:
            self.export_button.enabled = True
        else:
            self.export_button.enabled = False

    def _export_point_cloud(self, event):
        filename, _ = QFileDialog().getSaveFileName(
            self.native,
            "Save Point Cloud File...",
            "",
            "Orientations (*.tsv *.star);; All Files (*)",
        )

        if not filename:
            return None

        layer = self.viewer.layers.selection.active
        if layer and isinstance(layer, napari.layers.Points):
            original_dataframe = self.dataframes.get(
                layer.name,
                pd.DataFrame(columns=["x", "y", "z", "euler_x", "euler_y", "euler_z"]),
            )
            export_data = pd.DataFrame(layer.data, columns=["x", "y", "z"])
            merged_data = pd.merge(
                export_data, original_dataframe, on=["x", "y", "z"], how="left"
            )

            merged_data["z"] = merged_data["z"].astype(int)
            merged_data["y"] = merged_data["y"].astype(int)
            merged_data["x"] = merged_data["x"].astype(int)

            euler_columns = ["euler_z", "euler_y", "euler_x"]
            for col in euler_columns:
                if col not in merged_data.columns:
                    continue
                merged_data[col] = merged_data[col].fillna(0)

            if "score" in merged_data.columns:
                merged_data["score"] = merged_data["score"].fillna(1)
            merged_data["detail"] = layer.properties["detail"]
            merged_data = merged_data[layer.shown]

            translations = np.stack(
                (merged_data["x"], merged_data["y"], merged_data["z"]), axis=1
            )
            rotations = np.stack(
                (
                    merged_data["euler_x"],
                    merged_data["euler_y"],
                    merged_data["euler_z"],
                ),
                axis=1,
            )
            orientations = Orientations(
                translations=translations,
                rotations=rotations,
                scores=merged_data["score"],
                details=merged_data["detail"],
            )
            orientations.to_file(filename)

    def _get_load_path(self, event):
        filename, _ = QFileDialog().getOpenFileName(
            self.native,
            "Open Point Cloud File...",
            "",
            "Orientations (*.tsv *.star);; All Files (*)",
        )
        if filename:
            self._load_point_cloud(filename)

    def _load_point_cloud(self, filename):
        orientations = Orientations.from_file(filename)

        data = {
            "x": orientations.translations[:, 0],
            "y": orientations.translations[:, 1],
            "z": orientations.translations[:, 2],
            "euler_x": orientations.rotations[:, 0],
            "euler_y": orientations.rotations[:, 1],
            "euler_z": orientations.rotations[:, 2],
            "score": orientations.scores,
            "detail": orientations.details,
        }

        dataframe = pd.DataFrame.from_dict(data)
        layer_name = filename.split("/")[-1]

        if "score" not in dataframe.columns:
            dataframe["score"] = 1

        if "detail" not in dataframe.columns:
            dataframe["detail"] = -1

        point_properties = {
            "score": np.array(dataframe["score"].values),
            "detail": np.array(dataframe["detail"].values),
        }
        point_properties["score_scaled"] = np.log1p(
            point_properties["score"] - point_properties["score"].min()
        )
        self.score_range_slider.value = (0, 100)
        self.viewer.add_points(
            dataframe[["x", "y", "z"]].values,
            size=10,
            properties=point_properties,
            name=layer_name,
        )
        self.dataframes[layer_name] = dataframe


class MatchingWidget(widgets.Container):
    def __init__(self, viewer):
        super().__init__(layout="vertical")

        self.viewer = viewer
        self.dataframes = {}

        self.load_target_checkbox = widgets.CheckBox(text="Load Target", value=False)
        self.append(self.load_target_checkbox)
        self.import_button = widgets.PushButton(name="Import", text="Import Pickle")
        self.import_button.clicked.connect(self._get_load_path)

        self.append(self.import_button)

    def _get_load_path(self, event):
        filename, _ = QFileDialog.getOpenFileName(
            self.native,
            "Open Pickle File...",
            "",
            "Pickle Files (*.pickle);;All Files (*)",
        )
        if filename:
            self._load_data(filename)

    def _load_data(self, filename):
        data = load_pickle(filename)

        fname = basename(filename).replace(".pickle", "")

        if self.load_target_checkbox.value:
            try:
                target = Density.from_file(data[-1][-1].target)
                _ = self.viewer.add_image(
                    data=target.data,
                    name=f"{fname}_target",
                    metadata={
                        "origin": target.origin,
                        "sampling_rate": target.sampling_rate,
                    },
                )
            except Exception as e:
                msg = QMessageBox(self.native)
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Loading Error")
                msg.setText(str(e))
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()

        if data[0].ndim == data[2].ndim:
            metadata = {"origin": data[-1][1], "sampling_rate": data[-1][2]}
            _ = self.viewer.add_image(
                data=data[2],
                name=f"{fname}_rotations",
                colormap="orange",
                metadata=metadata,
            )
            _ = self.viewer.add_image(
                data=data[0],
                name=f"{fname}_scores",
                colormap="turbo",
                metadata=metadata,
            )
            return None

        point_properties = {"score": data[2], "detail": data[3]}
        point_properties["score_scaled"] = np.log1p(
            point_properties["score"] - point_properties["score"].min()
        )
        layer_name = f"{fname}_candidates"
        self.viewer.add_points(
            data[0],
            size=10,
            properties=point_properties,
            name=layer_name,
        )


class CustomNapariViewer(napari.Viewer):
    """
    Custom viewer to ensure 3D image layers are by default shown as xy projection.
    """

    def add_image(self, data, **kwargs):
        viewer_ndim = len(self.dims.order)
        layer = super().add_image(data, **kwargs)

        try:
            # Set to xy view the first time data is opened
            if viewer_ndim != 3 and data.ndim == 3:
                self.dims.order = (2, 0, 1)
        except Exception:
            pass
        return layer


def main():
    viewer = CustomNapariViewer()

    filter_widget = FilterWidget(preprocessor, viewer)
    mask_widget = MaskWidget(viewer)
    export_widget = ExportWidget(viewer)
    point_cloud = PointCloudWidget(viewer)
    matching_widget = MatchingWidget(viewer)
    alignment_widget = AlignmentWidget(viewer)

    viewer.window.add_dock_widget(widget=filter_widget, name="Preprocess", area="right")
    viewer.window.add_dock_widget(
        widget=alignment_widget, name="Alignment", area="right"
    )
    viewer.window.add_dock_widget(widget=mask_widget, name="Mask", area="right")
    viewer.window.add_dock_widget(widget=export_widget, name="Export", area="right")
    viewer.window.add_dock_widget(widget=point_cloud, name="PointCloud", area="left")
    viewer.window.add_dock_widget(widget=matching_widget, name="Matching", area="left")

    napari.run()


def parse_args():
    parser = argparse.ArgumentParser(
        description="GUI for preparing and analyzing template matching runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parse_args()
    main()
