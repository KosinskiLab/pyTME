#!python3
""" Simplify picking adequate filtering and masking parameters using a GUI.
    Exposes tme.preprocessor.Preprocessor and tme.fitter_utils member functions
    to achieve this aim.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import inspect
from typing import Tuple, Callable, List
from typing_extensions import Annotated

import numpy as np
import pandas as pd
import napari
from napari.layers import Image
from napari.utils.events import EventedList

from magicgui import widgets
from qtpy.QtWidgets import QFileDialog
from numpy.typing import NDArray

from tme import Preprocessor, Density
from tme.matching_utils import create_mask

preprocessor = Preprocessor()
SLIDER_MIN, SLIDER_MAX = 0, 25


def gaussian_filter(template, sigma: float, **kwargs: dict):
    return preprocessor.gaussian_filter(template=template, sigma=sigma, **kwargs)


def bandpass_filter(
    template,
    minimum_frequency: float,
    maximum_frequency: float,
    gaussian_sigma: float,
    **kwargs: dict,
):
    return preprocessor.bandpass_filter(
        template=template,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        sampling_rate=1,
        gaussian_sigma=gaussian_sigma,
        **kwargs,
    )


def difference_of_gaussian_filter(
    template, sigmas: Tuple[float, float], **kwargs: dict
):
    low_sigma, high_sigma = sigmas
    return preprocessor.difference_of_gaussian_filter(
        template=template, low_sigma=low_sigma, high_sigma=high_sigma, **kwargs
    )


def edge_gaussian_filter(
    template,
    sigma: float,
    edge_algorithm: Annotated[
        str,
        {"choices": ["sobel", "prewitt", "laplace", "gaussian", "gaussian_laplace"]},
    ],
    reverse: bool = False,
    **kwargs: dict,
):
    return preprocessor.edge_gaussian_filter(
        template=template,
        sigma=sigma,
        reverse=reverse,
        edge_algorithm=edge_algorithm,
    )


def local_gaussian_filter(
    template,
    lbd: float,
    sigma_range: Tuple[float, float],
    gaussian_sigma: float,
    reverse: bool = False,
    **kwargs: dict,
):
    return preprocessor.local_gaussian_filter(
        template=template,
        lbd=lbd,
        sigma_range=sigma_range,
        gaussian_sigma=gaussian_sigma,
    )


def ntree(
    template,
    sigma_range: Tuple[float, float],
    **kwargs: dict,
):
    return preprocessor.ntree_filter(template=template, sigma_range=sigma_range)


def mean(
    template,
    width: int,
    **kwargs: dict,
):
    return preprocessor.mean_filter(template=template, width=width)


def wedge(
    template: NDArray,
    tilt_start: float,
    tilt_stop: float,
    gaussian_sigma: float,
    tilt_axis: int = 1,
    infinite_plane : bool = True,
    extrude_plane : bool = True
):
    template_ft = np.fft.rfftn(template)
    wedge_mask = preprocessor.continuous_wedge_mask(
        start_tilt=tilt_start,
        stop_tilt=tilt_stop,
        tilt_axis=tilt_axis,
        shape=template.shape,
        sigma=gaussian_sigma,
        omit_negative_frequencies=True,
        infinite_plane = infinite_plane,
        extrude_plane=extrude_plane
    )
    np.multiply(template_ft, wedge_mask, out=template_ft)
    template = np.real(np.fft.irfftn(template_ft))
    return template


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
                value=param.default
                if param.default != inspect._empty
                else (0.0, SLIDER_MAX / 2),
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
                    value=param.default
                    if param.default != inspect._empty
                    else metadata["choices"][0],
                )
        else:
            continue
        ret.append(widget)
    return ret


WRAPPED_FUNCTIONS = {
    "gaussian_filter": gaussian_filter,
    "bandpass_filter": bandpass_filter,
    "edge_gaussian_filter": edge_gaussian_filter,
    "ntree_filter": ntree,
    "local_gaussian_filter": local_gaussian_filter,
    "difference_of_gaussian_filter": difference_of_gaussian_filter,
    "mean_filter" : mean,
    "continuous_wedge_mask" : wedge,
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
    # "continuous_wedge_mask",
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

        sanitized_names = [self._sanitize_name(name) for name in method_names]
        self.name_mapping.update(dict(zip(sanitized_names, method_names)))

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
            metadata["filter_parameters"].append({filter_name: kwargs.copy()})
            metadata["used_filter"] = filter_name
            new_layer.metadata = metadata


def sphere_mask(
    template: NDArray, center_x: float, center_y: float, center_z: float, radius: float
):
    return create_mask(
        mask_type="ellipse",
        shape=template.shape,
        center=(center_x, center_y, center_z),
        radius=radius,
    )


def ellipsod_mask(
    template: NDArray,
    center_x: float,
    center_y: float,
    center_z: float,
    radius_x: float,
    radius_y: float,
    radius_z: float,
):
    return create_mask(
        mask_type="ellipse",
        shape=template.shape,
        center=(center_x, center_y, center_z),
        radius=(radius_x, radius_y, radius_z),
    )


def box_mask(
    template: NDArray,
    center_x: float,
    center_y: float,
    center_z: float,
    height_x: int,
    height_y: int,
    height_z: int,
):
    return create_mask(
        mask_type="box",
        shape=template.shape,
        center=(center_x, center_y, center_z),
        height=(height_x, height_y, height_z),
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
):
    return create_mask(
        mask_type="tube",
        shape=template.shape,
        symmetry_axis=symmetry_axis,
        base_center=(center_x, center_y, center_z),
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        height=height,
    )


def wedge_mask(
    template: NDArray,
    tilt_start: float,
    tilt_stop: float,
    gaussian_sigma: float,
    tilt_axis: int = 1,
    omit_negative_frequencies : bool = True,
    extrude_plane : bool = True,
    infinite_plane : bool = True
):
    wedge_mask = preprocessor.continuous_wedge_mask(
        start_tilt=tilt_start,
        stop_tilt=tilt_stop,
        tilt_axis=tilt_axis,
        shape=template.shape,
        sigma=gaussian_sigma,
        omit_negative_frequencies=omit_negative_frequencies,
        extrude_plane=extrude_plane,
        infinite_plane=infinite_plane
    )
    wedge_mask = np.fft.fftshift(wedge_mask)
    return wedge_mask


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
            "Wedge": wedge_mask,
        }

        self.method_dropdown = widgets.ComboBox(
            name="Choose Mask", choices=list(self.methods.keys())
        )
        self.method_dropdown.changed.connect(self._on_method_changed)

        self.adapt_button = widgets.PushButton(
            text="Adapt to layer", enabled=False
        )
        self.adapt_button.changed.connect(self._update_initial_values)

        self.viewer.layers.selection.events.active.connect(
            self._update_action_button_state
        )

        self.align_button = widgets.PushButton(text="Align to axis", enabled=False)
        self.align_button.changed.connect(self._align_with_axis)
        self.density_field = widgets.Label()
        # self.density_field.value = f"Positive Density in Mask: {0:.2f}%"


        self.append(self.method_dropdown)
        self.append(self.adapt_button)
        self.append(self.align_button)
        self.append(self.action_button)
        self.append(self.density_field)

        # Create GUI for initially selected filtering method
        self._on_method_changed(None)

    def _update_action_button_state(self, event):
        self.align_button.enabled = bool(self.viewer.layers.selection.active)
        self.action_button.enabled = bool(self.viewer.layers.selection.active)
        self.adapt_button.enabled = bool(self.viewer.layers.selection.active)


    def _align_with_axis(self):
        active_layer = self.viewer.layers.selection.active

        if active_layer.metadata.get("is_aligned", False):
            return

        coords = np.array(np.where(active_layer.data > 0)).T
        centered_coords = coords - np.mean(coords, axis=0)
        cov_matrix = np.cov(centered_coords, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        principal_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]


        rotation_axis = np.cross(principal_eigenvector, [1, 0, 0])
        rotation_angle = np.arccos(np.dot(principal_eigenvector, [1, 0, 0]))
        k = rotation_axis / np.linalg.norm(rotation_axis)
        K = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ])
        rotation_matrix = np.eye(3)
        rotation_matrix += np.sin(rotation_angle) * K
        rotation_matrix += (1 - np.cos(rotation_angle)) * np.dot(K, K)

        rotated_data = Density.rotate_array(
            arr=active_layer.data,
            rotation_matrix=rotation_matrix,
            use_geometric_center=False
        )
        eps = np.finfo(rotated_data.dtype).eps
        rotated_data[rotated_data < eps] = 0

        active_layer.metadata["is_aligned"] = True
        active_layer.data = rotated_data


    def _update_initial_values(self, event=None):
        active_layer = self.viewer.layers.selection.active
        center_of_mass = Density.center_of_mass(np.abs(active_layer.data), 0)
        coordinates = np.array(np.where(active_layer.data > 0))
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

        defaults["radius"] = np.min(coordinate_radius)
        defaults["inner_radius"] = np.min(coordinate_radius)
        defaults["outer_radius"] = np.max(coordinate_radius)
        defaults["height"] = defaults["radius"]

        for widget in self.action_widgets:
            if widget.name in defaults:
                widget.value = defaults[widget.name]

    def _on_method_changed(self, event=None):
        for widget in self.action_widgets:
            self.remove(widget)
        self.action_widgets.clear()

        function = self.methods.get(self.method_dropdown.value)
        widgets = widgets_from_function(function)
        for widget in widgets:
            self.action_widgets.append(widget)
            self.insert(1, widget)

    def _action(self):
        function = self.methods.get(self.method_dropdown.value)

        selected_layer = self.viewer.layers.selection.active
        kwargs = {widget.name: widget.value for widget in self.action_widgets}
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

        origin_layer = metadata["origin_layer"]
        if origin_layer in self.viewer.layers:
            origin_layer = self.viewer.layers[origin_layer]
            if np.allclose(origin_layer.data.shape, processed_data.shape):
                in_mask = np.sum(np.fmax(origin_layer.data * processed_data, 0))
                in_mask /= np.sum(np.fmax(origin_layer.data, 0))
                in_mask *= 100
                self.density_field.value = f"Positive Density in Mask: {in_mask:.2f}%"




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
        options = QFileDialog.Options()
        path, _ = QFileDialog.getSaveFileName(
            self.native,
            "Save As...",
            "",
            "MRC Files (*.mrc)",
            options=options,
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

        self.import_button = widgets.PushButton(
            name="Import", text="Import Point Cloud"
        )
        self.import_button.clicked.connect(self._get_load_path)

        self.export_button = widgets.PushButton(
            name="Export", text="Export Point Cloud"
        )
        self.export_button.clicked.connect(self._export_point_cloud)
        self.export_button.enabled = False

        self.append(self.import_button)
        self.append(self.export_button)
        self.viewer.layers.selection.events.changed.connect(self._update_buttons)

    def _update_buttons(self, event):
        is_pointcloud = isinstance(
            self.viewer.layers.selection.active, napari.layers.Points
        )
        if self.viewer.layers.selection.active and is_pointcloud:
            self.export_button.enabled = True
        else:
            self.export_button.enabled = False

    def _export_point_cloud(self, event):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self.native,
            "Save Point Cloud File...",
            "",
            "TSV Files (*.tsv);;All Files (*)",
            options=options,
        )

        if filename:
            layer = self.viewer.layers.selection.active
            if layer and isinstance(layer, napari.layers.Points):
                original_dataframe = self.dataframes.get(layer.name, pd.DataFrame())

                export_data = pd.DataFrame(layer.data, columns=["z", "y", "x"])
                merged_data = pd.merge(
                    export_data, original_dataframe, on=["z", "y", "x"], how="left"
                )
                merged_data.to_csv(filename, sep="\t", index=False)

    def _get_load_path(self, event):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self.native,
            "Open Point Cloud File...",
            "",
            "TSV Files (*.tsv);;All Files (*)",
            options=options,
        )
        if filename:
            self._load_point_cloud(filename)

    def _load_point_cloud(self, filename):
        dataframe = pd.read_csv(filename, sep="\t")
        points = dataframe[["z", "y", "x"]].values
        layer_name = filename.split("/")[-1]
        self.viewer.add_points(points, size=10, name=layer_name)
        self.dataframes[layer_name] = dataframe


def main():
    viewer = napari.Viewer()

    filter_widget = FilterWidget(preprocessor, viewer)
    mask_widget = MaskWidget(viewer)
    export_widget = ExportWidget(viewer)
    point_cloud = PointCloudWidget(viewer)

    viewer.window.add_dock_widget(widget=filter_widget, name="Preprocess", area="right")
    viewer.window.add_dock_widget(widget=mask_widget, name="Mask", area="right")
    viewer.window.add_dock_widget(widget=point_cloud, name="PointCloud", area="left")

    viewer.window.add_dock_widget(widget=export_widget, name="Export", area="right")

    napari.run()


if __name__ == "__main__":
    main()
