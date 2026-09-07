#!python3
"""
GUI for identifying suitable masks and analyzing template matchign results.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import argparse
from os.path import basename

import napari
import numpy as np
from magicgui import widgets
from numpy.typing import NDArray
from napari.layers import Image
from qtpy.QtWidgets import QFileDialog, QMessageBox
from scipy.ndimage import gaussian_filter

from tme.backends import backend as be
from tme.rotations import align_vectors
from tme import Density
from tme.utils.serialization import deserialize
from tme.rotations import euler_from_rotationmatrix
from tme.matching_utils import create_mask, center_slice
from tme.filters import BandPassReconstructed, WedgeReconstructed

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from tme.matching_utils import minimum_score_from_fp
from tme.scripts.postprocess import PEAK_CALLERS

from napari.utils.theme import get_theme
from napari.qt.threading import thread_worker


def _apply_napari_theme(figure, axes, theme):
    """Style a Matplotlib figure to match a napari Theme."""
    fields = {}
    for name in (
        "background",
        "foreground",
        "text",
        "secondary",
        "highlight",
        "canvas",
    ):
        value = getattr(theme, name, None)
        if value is None:
            continue
        fields[name] = str(value)

    bg = fields.get("background")
    canvas_color = fields.get("canvas", bg)
    fg = fields.get("foreground")
    text = fields.get("text", fg)
    secondary = fields.get("secondary", fg)
    highlight = fields.get("highlight", fg)

    if bg is not None:
        figure.patch.set_facecolor(bg)
    if canvas_color is not None:
        axes.set_facecolor(canvas_color)
    for spine in axes.spines.values():
        if fg is not None:
            spine.set_color(fg)
    if fg is not None:
        axes.tick_params(colors=fg)
    if text is not None:
        axes.xaxis.label.set_color(text)
        axes.yaxis.label.set_color(text)
        for label in axes.get_xticklabels() + axes.get_yticklabels():
            label.set_color(text)

    return {
        "bar_face": secondary,
        "bar_edge": highlight,
        "threshold": highlight,
    }


def _apply_fourier_filter(arr, arr_filter):
    arr_ft = np.fft.rfftn(arr, s=arr.shape)
    arr_ft = np.multiply(arr_ft, arr_filter, out=arr_ft)
    return np.real(np.fft.irfftn(arr_ft, s=arr.shape))


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
    )(shape=template.shape, return_real_fourier=True)["data"]
    return _apply_fourier_filter(template, bpf)


def _mask_param_widgets(function) -> list:
    """Build a magicgui widget per non-skipped parameter of `function`."""
    from inspect import signature, Parameter

    skip = {"self", "template", "shapes_layer", "kwargs"}
    widgets_out = []
    for name, param in signature(function).parameters.items():
        if name in skip or param.kind is Parameter.VAR_KEYWORD:
            continue
        default = param.default if param.default is not Parameter.empty else 0
        if param.annotation is int:
            w = widgets.SpinBox(name=name, value=default, min=0, max=10_000)
        else:
            w = widgets.FloatSpinBox(
                name=name, value=default, min=0, max=1_000_000, step=0.5
            )
        widgets_out.append(w)
    return widgets_out


def sphere_mask(
    template: NDArray,
    center_x: float,
    center_y: float,
    center_z: float,
    radius: float,
    soft_edge_width: float = 0,
    **kwargs,
) -> NDArray:
    return create_mask(
        mask_type="ellipse",
        shape=template.shape,
        center=(center_x, center_y, center_z),
        radius=radius,
        soft_edge_width=soft_edge_width,
    )


def ellipsod_mask(
    template: NDArray,
    center_x: float,
    center_y: float,
    center_z: float,
    radius_x: float,
    radius_y: float,
    radius_z: float,
    soft_edge_width: float = 0,
    **kwargs,
) -> NDArray:
    return create_mask(
        mask_type="ellipse",
        shape=template.shape,
        center=(center_x, center_y, center_z),
        radius=(radius_x, radius_y, radius_z),
        soft_edge_width=soft_edge_width,
    )


def box_mask(
    template: NDArray,
    center_x: float,
    center_y: float,
    center_z: float,
    height_x: int,
    height_y: int,
    height_z: int,
    soft_edge_width: float = 0,
    **kwargs,
) -> NDArray:
    return create_mask(
        mask_type="box",
        shape=template.shape,
        center=(center_x, center_y, center_z),
        size=(height_x, height_y, height_z),
        soft_edge_width=soft_edge_width,
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
    soft_edge_width: float = 0,
    **kwargs,
) -> NDArray:
    return create_mask(
        center=(center_x, center_y, center_z),
        mask_type="membrane",
        shape=template.shape,
        radius=radius,
        thickness=thickness,
        separation=separation,
        soft_edge_width=soft_edge_width,
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
    soft_edge_width: float = 0,
    **kwargs,
) -> NDArray:
    return create_mask(
        mask_type="tube",
        shape=template.shape,
        symmetry_axis=symmetry_axis,
        center=(center_x, center_y, center_z),
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        height=height,
        soft_edge_width=soft_edge_width,
    )


def wedge_mask(
    template: NDArray,
    tilt_start: float = 40.0,
    tilt_stop: float = 40.0,
    tilt_step: float = 0,
    opening_axis: int = 2,
    tilt_axis: int = 0,
    **kwargs,
) -> NDArray:
    angles = (tilt_start, tilt_stop)
    continuous_wedge = tilt_step == 0
    if not continuous_wedge:
        angles = np.arange(-tilt_start, tilt_stop + tilt_step, tilt_step)

    return_real_fourier = kwargs.get("return_real_fourier", False)
    func = WedgeReconstructed(
        angles=angles,
        tilt_axis=tilt_axis,
        opening_axis=opening_axis,
        frequency_cutoff=0.5,
        create_continuous_wedge=continuous_wedge,
        weight_wedge=kwargs.get("weight_angle", False),
    )
    wedge_mask = func(shape=template.shape, return_real_fourier=return_real_fourier)[
        "data"
    ]
    if kwargs.get("fftshift", True):
        axes = [i for i in range(wedge_mask.ndim)]
        if return_real_fourier:
            _ = axes.pop(-1)
        wedge_mask = np.fft.fftshift(wedge_mask, axes=axes)
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
        mask_filter = gaussian_filter(mask.astype(float), sigma=sigma)
        mask = np.add(mask, (1 - mask) * mask_filter)
        mask[mask < np.exp(-np.square(sigma))] = 0

    if invert:
        mask = 1 - mask
    return mask


def lowpass_mask(
    template: NDArray,
    lowpass_angstrom: float = 30.0,
    soft_edge_width: float = 0,
    x_stretch: float = 100.0,
    y_stretch: float = 100.0,
    z_stretch: float = 100.0,
    **kwargs,
):
    from scipy.ndimage import zoom
    from skimage.filters import threshold_otsu

    smoothed = bandpass_filter(
        template,
        lowpass_angstrom=lowpass_angstrom,
        highpass_angstrom=None,
        sampling_rate=kwargs.get("sampling_rate", 1),
    )
    mask = smoothed > threshold_otsu(smoothed)

    stretch_factors = (x_stretch / 100, y_stretch / 100, z_stretch / 100)
    if any(factor != 1.0 for factor in stretch_factors):

        original_shape = mask.shape
        mask = zoom(mask.astype(float), stretch_factors, order=1)

        subset = center_slice(mask.shape, new_shape=original_shape)
        mask = mask[subset] > 0.5

    if soft_edge_width != 0:
        mask_filter = gaussian_filter(mask.astype(float), sigma=soft_edge_width)
        mask_filter /= mask_filter.max()
        mask = np.add(mask, (1 - mask) * mask_filter)
        mask[mask < np.exp(-np.square(soft_edge_width))] = 0

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

        self.preview_button = widgets.PushButton(
            text="Preview on template", enabled=False
        )
        self.preview_button.changed.connect(self._preview_on_template)

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
        self.append(self.preview_button)
        self.append(self.action_button)
        self.append(self.density_field)

        # Create GUI for initially selected filtering method
        self._on_method_changed(None)

    def _update_action_button_state(self, event):
        self.action_button.enabled = bool(self.viewer.layers.selection.active)
        self.adapt_button.enabled = bool(self.viewer.layers.selection.active)
        self.preview_button.enabled = bool(self.viewer.layers.selection.active)

    def _update_initial_values(self, event=None):
        active_layer = self.viewer.layers.selection.active

        data = active_layer.data

        # Threshold at the 0.99 quantile (the EMDB-style contour level) so the
        # adapted geometry tracks the dense core, not stray low-intensity voxels.
        contour_level = float(np.quantile(np.abs(data), 0.99))
        center_of_mass = Density.center_of_mass(np.abs(data), contour_level)
        coordinates = np.array(np.where(np.abs(data) >= contour_level))
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
        function_widgets = _mask_param_widgets(function)
        for widget in function_widgets:
            self.action_widgets.append(widget)
            self.insert(1, widget)

        from inspect import signature

        if "shapes_layer" in signature(function).parameters:
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
        method = self.method_dropdown.value
        function = self.methods.get(method)

        source_layer = self.viewer.layers.selection.active
        kwargs = {widget.name: widget.value for widget in self.action_widgets}

        if "shapes_layer" in kwargs:
            layer_name = kwargs["shapes_layer"]
            if layer_name not in self.viewer.layers:
                return None
            kwargs["shapes_layer"] = self.viewer.layers[layer_name]
            kwargs["expansion_dim"] = self.viewer.dims.order[0]

        try:
            sampling = source_layer.metadata["sampling_rate"]
        except Exception:
            sampling = 1

        processed_data = function(
            template=source_layer.data, sampling_rate=sampling, **kwargs
        ).astype(np.float32)

        new_layer_name = f"{source_layer.name} ({method})"

        if new_layer_name in self.viewer.layers:
            target = self.viewer.layers[new_layer_name]
            target.data = processed_data
            base_metadata = dict(target.metadata)
        else:
            target = self.viewer.add_image(data=processed_data, name=new_layer_name)
            base_metadata = {
                k: source_layer.metadata[k]
                for k in ("sampling_rate", "origin")
                if k in source_layer.metadata
            }

        if method == "Shape":
            target.metadata = {}
            return

        # The napari-pytme-io writer dumps filter_parameters verbatim into the
        # sidecar YAML; drop the live Shapes layer reference so what gets recorded
        # is the values a user would re-supply, not Python objects.
        stored_kwargs = {n: v for n, v in kwargs.items() if n != "shapes_layer"}

        base_metadata["mask"] = method
        base_metadata["origin_layer"] = source_layer.name
        base_metadata["filter_parameters"] = {method: stored_kwargs}
        target.metadata = base_metadata

    def _preview_on_template(self):
        mask = self.viewer.layers.selection.active
        if mask is None:
            return
        origin_name = mask.metadata.get("origin_layer")
        if not origin_name or origin_name not in self.viewer.layers:
            return
        template = self.viewer.layers[origin_name]
        if template.data.shape != mask.data.shape:
            return
        self.viewer.add_image(
            template.data * mask.data,
            name=f"{template.name} (masked)",
        )


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
        rotated_data, _ = be.rigid_transform(
            arr=active_layer.data,
            rotation_matrix=rotation_matrix,
            center="geometric",
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

        horizontal_container = widgets.Container(layout="horizontal")

        self.gzip_output = widgets.CheckBox(name="gzip", value=False, label="gzip")
        self.export_button = widgets.PushButton(name="Export", text="Export")
        self.export_button.clicked.connect(self._export)

        horizontal_container.append(self.export_button)
        horizontal_container.append(self.gzip_output)

        self.append(horizontal_container)

        self._update_export_button_state(None)
        self.viewer.layers.selection.events.active.connect(
            self._update_export_button_state
        )

    def _update_export_button_state(self, _event):
        layer = self.viewer.layers.selection.active
        self.export_button.enabled = isinstance(layer, Image)

    def _export(self):
        layer = self.viewer.layers.selection.active
        if isinstance(layer, Image):
            self._export_image(layer)

    def _export_image(self, layer):
        path, _ = QFileDialog().getSaveFileName(
            self.native,
            "Save As...",
            "",
            "MRC Files (*.mrc)",
        )
        if not path:
            return
        layer.metadata["write_gzip"] = self.gzip_output.value
        layer.save(path=path)


class CandidatesWidget(widgets.Container):
    def __init__(self, viewer):
        super().__init__(layout="vertical")
        self.viewer = viewer

        self.peak_caller = widgets.ComboBox(
            name="Peak caller",
            choices=("PeakCallerMaximumFilter", "PeakCallerRecursiveMasking"),
            value="PeakCallerMaximumFilter",
        )
        self.num_peaks = widgets.SpinBox(
            name="num_peaks", value=1000, min=1, max=1_000_000
        )
        self.min_distance = widgets.SpinBox(
            name="min_distance", value=10, min=0, max=10_000
        )
        self.min_boundary = widgets.SpinBox(
            name="min_boundary_distance", value=0, min=0, max=10_000
        )
        self.run_peak_call_button = widgets.PushButton(
            text="Run peak calling", enabled=False
        )
        self.run_peak_call_button.clicked.connect(self._run_peak_calling)

        # Threshold state — recomputed on selection change.
        self._scores_std = 0.0
        self._n_correlations = None
        self._suppress_writeback = False

        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QSlider, QLabel, QFrame

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setValue(0)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._slider_label = QLabel("min score: -")

        self.n_fp = widgets.FloatSpinBox(
            name="Auto from N_fp", value=10.0, min=0, max=1e15, step=1.0
        )
        self.n_fp.enabled = False
        self.n_fp.changed.connect(self._on_n_fp_changed)

        self._figure = Figure(figsize=(4, 2), constrained_layout=True)
        self._axes = self._figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self._figure)
        self.canvas.setMinimumHeight(140)
        self._theme_colors = {}
        self._threshold_line = None
        self._refresh_theme()

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)

        for w in (
            self.peak_caller,
            self.num_peaks,
            self.min_distance,
            self.min_boundary,
            self.run_peak_call_button,
        ):
            self.append(w)
        # Embed Qt widgets via magicgui's native bridge.
        self.native.layout().addWidget(separator)
        self.native.layout().addWidget(self.canvas)
        self.native.layout().addWidget(self._slider_label)
        self.native.layout().addWidget(self._slider)
        self.append(self.n_fp)

        self.viewer.events.theme.connect(self._on_theme_changed)

        self.viewer.layers.selection.events.changed.connect(self._on_selection_changed)

    def _refresh_theme(self):
        theme = get_theme(self.viewer.theme)
        self._theme_colors = _apply_napari_theme(self._figure, self._axes, theme)
        self.canvas.draw_idle()

    def _on_theme_changed(self, _event=None):
        self._refresh_theme()
        layer = self._active_points()
        if layer is not None and "score" in layer.properties:
            self._redraw_histogram(layer.properties["score"])

    def _on_slider_changed(self, _value=None):
        if self._suppress_writeback:
            return
        layer = self._active_points()
        if layer is None or "score" not in layer.properties:
            return
        threshold = self._threshold_from_slider(layer)
        self._set_threshold(threshold, source="slider")

    def _threshold_from_slider(self, layer):
        scores = np.asarray(layer.properties["score"])
        lo, hi = float(scores.min()), float(scores.max())
        if hi == lo:
            return lo
        return lo + (hi - lo) * (self._slider.value() / 1000.0)

    def _slider_value_from_threshold(self, layer, threshold):
        scores = np.asarray(layer.properties["score"])
        lo, hi = float(scores.min()), float(scores.max())
        if hi == lo:
            return 0
        frac = (float(threshold) - lo) / (hi - lo)
        return int(round(max(0.0, min(1.0, frac)) * 1000))

    def _set_threshold(self, threshold, source):
        """Single update path: drives layer.shown, the histogram line, and the
        N_fp readout when the slider was the driver."""
        layer = self._active_points()
        if layer is None or "score" not in layer.properties:
            return

        scores = np.asarray(layer.properties["score"])
        layer.shown = scores >= float(threshold)

        if self._threshold_line is not None:
            self._threshold_line.set_xdata([float(threshold), float(threshold)])
            self.canvas.draw_idle()

        self._slider_label.setText(f"min score: {float(threshold):.4g}")

        if (
            source == "slider"
            and self._n_correlations is not None
            and self._scores_std > 0
        ):
            from scipy.special import erfc

            ratio = float(threshold) / (np.sqrt(2.0) * self._scores_std)
            n_fp = 0.5 * self._n_correlations * float(erfc(ratio))
            if not np.isfinite(n_fp):
                return
            n_fp = max(0.0, min(float(self.n_fp.max), n_fp))
            self._suppress_writeback = True
            try:
                self.n_fp.value = n_fp
            finally:
                self._suppress_writeback = False

    def _active_points(self):
        layer = self.viewer.layers.selection.active
        return layer if isinstance(layer, napari.layers.Points) else None

    def _active_scores(self):
        layer = self.viewer.layers.selection.active
        if isinstance(layer, napari.layers.Image) and layer.name.endswith("_scores"):
            return layer
        return None

    def _on_selection_changed(self, _event=None):
        pts = self._active_points()
        scores_layer = self._active_scores()

        self.run_peak_call_button.enabled = scores_layer is not None

        has_score = pts is not None and "score" in pts.properties
        self._slider.setEnabled(has_score)
        self.n_fp.enabled = False  # re-enabled below if metadata supports it

        if not has_score:
            return

        scores = np.asarray(pts.properties["score"])
        # Rickgauer's N_fp formula needs the std of the full score *volume*, not
        # the std of the surviving peaks. Prefer the value stored at peak-call
        # time; only fall back to the peak-score std when nothing else is known
        # (e.g. orientations loaded from disk without provenance).
        volume_std = pts.metadata.get("scores_volume_std")
        if volume_std is None:
            volume_std = float(np.std(scores))
        self._scores_std = float(volume_std)
        self._n_correlations = pts.metadata.get("n_correlations")
        self.n_fp.enabled = self._n_correlations is not None and self._scores_std > 0

        # Clamp slider to a sane initial position: show all points.
        self._suppress_writeback = True
        try:
            self._slider.setValue(0)
        finally:
            self._suppress_writeback = False

        self._redraw_histogram(scores)
        # Apply the initial threshold (= score.min()) so layer.shown is well-defined.
        self._set_threshold(float(scores.min()), source="selection")

    def _redraw_histogram(self, scores):
        self._axes.clear()
        bar_face = self._theme_colors.get("bar_face") or "#888888"
        bar_edge = self._theme_colors.get("bar_edge") or "#cccccc"
        threshold_color = self._theme_colors.get("threshold") or "red"

        self._axes.hist(
            np.asarray(scores),
            bins=64,
            facecolor=bar_face,
            edgecolor=bar_edge,
        )
        self._threshold_line = self._axes.axvline(
            float(np.min(scores)),
            color=threshold_color,
            linestyle="--",
        )
        self._axes.set_xlabel("score")
        self.canvas.draw_idle()

    def _resolve_mask(self, scores_layer):
        mask_path = scores_layer.metadata.get("template_mask_path")
        if not mask_path:
            return None
        try:
            return Density.from_file(mask_path).data
        except Exception as exc:
            raise RuntimeError(
                f"Could not load template mask from '{mask_path}': {exc}. "
                "If you copied the scores file from another machine, also copy the "
                "template mask, or pick a peak caller that does not require one."
            ) from exc

    def _run_peak_calling(self):
        scores_layer = self._active_scores()
        if scores_layer is None:
            return None
        rot_name = scores_layer.name.replace("_scores", "_rotations")
        rot_layer = (
            self.viewer.layers[rot_name] if rot_name in self.viewer.layers else None
        )
        rotations = (
            rot_layer.data
            if rot_layer is not None
            else np.zeros(scores_layer.data.shape, dtype=int)
        )
        rotation_mapping = scores_layer.metadata.get(
            "rotation_mapping", {0: np.eye(scores_layer.data.ndim)}
        )

        caller_name = self.peak_caller.value
        needs_mask = caller_name == "PeakCallerRecursiveMasking"
        try:
            mask = self._resolve_mask(scores_layer)
        except RuntimeError as exc:
            if needs_mask:
                QMessageBox.critical(self.native, "Template mask required", str(exc))
                return None
            mask = None
        if mask is None and needs_mask:
            QMessageBox.critical(
                self.native,
                "Template mask required",
                "PeakCallerRecursiveMasking requires a template mask, but the scores "
                "layer does not record one. Re-run from a directory where the original "
                "template mask is accessible, or pick a different peak caller.",
            )
            return None

        peak_kwargs = dict(
            shape=scores_layer.data.shape,
            num_peaks=int(self.num_peaks.value),
            min_distance=int(self.min_distance.value),
            min_boundary_distance=int(self.min_boundary.value),
            min_score=None,
            max_score=None,
            batch_dims=None,
            projection_dims=None,
        )
        call_kwargs = dict(
            rotation_mapping=rotation_mapping,
            rotations=rotations,
            rotation_matrix=np.eye(scores_layer.data.ndim),
        )
        if mask is not None:
            call_kwargs["mask"] = mask

        scores_layer_name = scores_layer.name
        scores_data = scores_layer.data
        n_correlations = int(np.size(scores_data) * len(rotation_mapping))
        scores_volume_std = scores_layer.metadata.get("scores_volume_std")
        if scores_volume_std is None:
            scores_volume_std = float(np.std(scores_data))

        self.run_peak_call_button.enabled = False

        @thread_worker(progress={"total": 0})
        def _do_peak_call():
            caller = PEAK_CALLERS[caller_name](**peak_kwargs)
            state = caller.init_state()
            state = caller(state, scores_data, **call_kwargs)
            return caller.merge(results=[caller.result(state)], **peak_kwargs)

        def _on_returned(result):
            self.run_peak_call_button.enabled = True
            self._materialize_peak_call_result(
                result, scores_layer_name, n_correlations, scores_volume_std
            )

        def _on_errored(exc):
            self.run_peak_call_button.enabled = True
            QMessageBox.critical(self.native, "Peak calling failed", str(exc))

        worker = _do_peak_call()
        worker.returned.connect(_on_returned)
        worker.errored.connect(_on_errored)
        worker.start()

    def _materialize_peak_call_result(
        self, result, scores_layer_name, n_correlations, scores_volume_std
    ):
        translations, rotations_mat, peak_scores, _details = result
        translations = np.asarray(translations)
        peak_scores = np.asarray(peak_scores, dtype=np.float32)
        n = translations.shape[0]

        if rotations_mat is None or len(rotations_mat) == 0:
            eulers = np.zeros((n, 3), dtype=np.float32)
        else:
            eulers = np.stack(
                [euler_from_rotationmatrix(np.asarray(r)) for r in rotations_mat]
            ).astype(np.float32)

        properties = {
            "score": peak_scores,
            "euler_x": eulers[:, 0],
            "euler_y": eulers[:, 1],
            "euler_z": eulers[:, 2],
        }

        # Pass metadata via kwarg so the selection-changed event napari fires
        # from add_points sees a fully-populated layer; otherwise N_fp stays
        # disabled until the user clicks away and back.
        metadata = {
            "origin_scores_layer": scores_layer_name,
            "n_correlations": n_correlations,
            "scores_volume_std": float(scores_volume_std),
            "optics": {},
        }
        self.viewer.add_points(
            translations,
            size=10,
            properties=properties,
            face_color="score",
            face_colormap="turbo",
            name=f"{scores_layer_name} (candidates)",
            metadata=metadata,
        )

    def _on_n_fp_changed(self, _value=None):
        if self._suppress_writeback:
            return
        layer = self._active_points()
        if layer is None or "score" not in layer.properties:
            return
        if self._n_correlations is None or self._scores_std <= 0:
            return
        threshold = minimum_score_from_fp(
            self._scores_std, self._n_correlations, float(self.n_fp.value)
        )
        new_slider_value = self._slider_value_from_threshold(layer, threshold)
        self._suppress_writeback = True
        try:
            self._slider.setValue(new_slider_value)
        finally:
            self._suppress_writeback = False
        self._set_threshold(threshold, source="n_fp")


class MatchingWidget(widgets.Container):
    def __init__(self, viewer):
        super().__init__(layout="vertical")

        self.viewer = viewer
        self.dataframes = {}

        option_container = widgets.Container(layout="horizontal")
        self.load_target_checkbox = widgets.CheckBox(text="Load Target", value=False)
        self.load_rotations_checkbox = widgets.CheckBox(
            text="Load Rotations", value=False
        )
        option_container.append(self.load_target_checkbox)
        option_container.append(self.load_rotations_checkbox)

        self.import_button = widgets.PushButton(name="Import", text="Import Pickle")
        self.import_button.clicked.connect(self._get_load_path)

        self.append(option_container)
        self.append(self.import_button)

    def _get_load_path(self, event):
        filename, _ = QFileDialog.getOpenFileName(
            self.native,
            "Open Pickle File...",
            "",
            "Matching Files (*.pickle *pickle.gz *hdf5 *h5);;All Files (*)",
        )
        if filename:
            self._load_data(filename)

    def _load_data(self, filename):
        load_target = self.load_target_checkbox.value
        load_rotations = self.load_rotations_checkbox.value

        self.import_button.enabled = False

        @thread_worker(progress={"total": 0})
        def _do_load():
            data = deserialize(filename)
            target_payload = None
            if load_target:
                target = Density.from_file(data[-1][-1].target)
                target_payload = (
                    target.data,
                    target.origin,
                    target.sampling_rate,
                )
            return data, target_payload

        def _on_returned(result):
            self.import_button.enabled = True
            data, target_payload = result
            fname = basename(filename).replace(".pickle", "")
            if target_payload is not None:
                t_data, t_origin, t_sampling = target_payload
                self.viewer.add_image(
                    data=t_data,
                    name=f"{fname}_target",
                    metadata={"origin": t_origin, "sampling_rate": t_sampling},
                )
            self._materialize_load(data, fname, load_rotations)

        def _on_errored(exc):
            self.import_button.enabled = True
            msg = QMessageBox(self.native)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Loading Error")
            msg.setText(str(exc))
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

        worker = _do_load()
        worker.returned.connect(_on_returned)
        worker.errored.connect(_on_errored)
        worker.start()

    def _materialize_load(self, data, fname, load_rotations):
        if data[0].ndim == data[2].ndim:
            rotation_mapping = data[3] if len(data) > 3 else {0: np.eye(data[0].ndim)}
            scores = data[0]
            scores[np.isneginf(scores)] = 0
            # Prefer the running variance recorded during matching (data[4] when
            # the pickle includes it) over the std of the saved scores volume
            # see tme/scripts/postprocess.py for the same convention.
            if len(data) == 6:
                scores_volume_std = float(np.sqrt(data[4]).reshape(()))
            else:
                scores_volume_std = float(np.std(scores))
            metadata = {
                "origin": data[-1][0],
                "sampling_rate": data[-1][2],
                "rotation_mapping": rotation_mapping,
                "n_correlations": int(np.size(scores) * len(rotation_mapping)),
                "scores_volume_std": scores_volume_std,
                "template_mask_path": getattr(data[-1][-1], "template_mask", None),
            }
            if load_rotations:
                self.viewer.add_image(
                    data=data[2],
                    name=f"{fname}_rotations",
                    colormap="orange",
                    metadata=metadata,
                )
            self.viewer.add_image(
                data=scores,
                name=f"{fname}_scores",
                colormap="turbo",
                metadata=metadata,
            )
            return

        # Candidates branch — populate the layer data contract.
        translations = np.asarray(data[0])
        n = translations.shape[0]
        score = np.asarray(data[2], dtype=np.float32)
        rotations_arr = data[1] if len(data) > 1 else None

        if rotations_arr is None:
            eulers = np.zeros((n, 3), dtype=np.float32)
        else:
            rotations_arr = np.asarray(rotations_arr)
            if rotations_arr.ndim == 2 and rotations_arr.shape[1] == 3:
                # Already eulers
                eulers = rotations_arr.astype(np.float32)
            elif rotations_arr.ndim == 3 and rotations_arr.shape[1:] == (3, 3):
                eulers = np.stack(
                    [euler_from_rotationmatrix(r) for r in rotations_arr]
                ).astype(np.float32)
            else:
                eulers = np.zeros((n, 3), dtype=np.float32)

        properties = {
            "score": score,
            "euler_x": eulers[:, 0],
            "euler_y": eulers[:, 1],
            "euler_z": eulers[:, 2],
            "_rlnClassNumber": np.asarray(data[3]) if len(data) > 3 else np.full(n, -1),
        }
        self.viewer.add_points(
            translations,
            size=10,
            properties=properties,
            face_color="score",
            face_colormap="turbo",
            name=f"{fname}_candidates",
            metadata={"optics": {}},
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


class ImageActionsWidget(widgets.Container):
    def __init__(self, viewer):
        super().__init__(layout="vertical")
        self.viewer = viewer
        self.power_spectrum_btn = widgets.PushButton(text="Power spectrum")
        self.invert_btn = widgets.PushButton(text="Invert contrast")
        self.power_spectrum_btn.clicked.connect(self._power_spectrum)
        self.invert_btn.clicked.connect(self._invert_contrast)
        container = widgets.Container(
            widgets=[self.power_spectrum_btn, self.invert_btn], layout="horizontal"
        )
        self.append(container)

    def _active_image(self):
        layer = self.viewer.layers.selection.active
        return layer if isinstance(layer, napari.layers.Image) else None

    def _power_spectrum(self):
        layer = self._active_image()
        if layer is None:
            return
        data = np.fft.fftshift(np.log1p(np.abs(np.fft.fftn(layer.data)) ** 2))
        self.viewer.add_image(
            data,
            name=f"{layer.name} (power spectrum)",
            metadata=dict(layer.metadata),
        )

    def _invert_contrast(self):
        layer = self._active_image()
        if layer is None:
            return
        self.viewer.add_image(
            layer.data * -1,
            name=f"{layer.name} (inverted)",
            metadata=dict(layer.metadata),
        )


def main():
    viewer = CustomNapariViewer()

    mask_widget = MaskWidget(viewer)
    image_actions = ImageActionsWidget(viewer)
    export_widget = ExportWidget(viewer)
    point_cloud = CandidatesWidget(viewer)
    matching_widget = MatchingWidget(viewer)
    alignment_widget = AlignmentWidget(viewer)

    viewer.window.add_dock_widget(
        widget=image_actions, name="Image actions", area="left"
    )
    viewer.window.add_dock_widget(
        widget=alignment_widget, name="Alignment", area="left"
    )
    viewer.window.add_dock_widget(widget=matching_widget, name="Matching", area="left")
    viewer.window.add_dock_widget(widget=mask_widget, name="Mask", area="right")
    viewer.window.add_dock_widget(widget=point_cloud, name="Candidates", area="right")
    viewer.window.add_dock_widget(widget=export_widget, name="Export", area="right")

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
