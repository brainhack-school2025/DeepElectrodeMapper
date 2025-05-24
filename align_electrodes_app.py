import sys
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import (QApplication, QWidget, QSlider, QVBoxLayout, QLabel,
                             QHBoxLayout, QPushButton, QFileDialog)
from PyQt5.QtCore import Qt
from scipy.spatial.transform import Rotation as R

# === Load files ===
electrode_file = "sourcedata/sub-010_ses-01_acq-structure_electrodes.txt"
obj_file = "sourcedata/sub-010_obj/model_mesh.obj"

coords = []
labels = []

with open(electrode_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        label, x, y, z = parts
        coords.append([float(x), float(y), float(z)])
        labels.append(label)

coords = np.array(coords) / 1000  # mm → meters

# Extract fiducials
fiducial_labels = ['nas', 'lhj', 'rhj']
fiducial_coords = np.array([coords[labels.index(lab)] for lab in fiducial_labels])
remaining_labels = [lab for lab in labels if lab not in fiducial_labels]
remaining_coords = np.array([coords[i] for i, lab in enumerate(labels) if lab not in fiducial_labels])


class ElectrodeAligner(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Electrode Alignment Tool")

        self.original_coords = remaining_coords.copy()
        self.transformed_coords = self.original_coords.copy()
        self.fiducial_coords = fiducial_coords.copy()
        self.surface_fiducials = []

        self.plotter = BackgroundPlotter(show=True)
        self.mesh = pv.read(obj_file)
        self.plotter.add_mesh(self.mesh, color="lightgray", opacity=0.8)
        self.add_axes()

        self.glyph_actor = None

        self.init_ui()
        self.update_plot()

    def add_axes(self):
        arrow_length = 0.05
        origin = np.array([0, 0, 0])
        x_arrow = pv.Arrow(start=origin, direction=[1, 0, 0], tip_length=0.3, scale=arrow_length)
        y_arrow = pv.Arrow(start=origin, direction=[0, 1, 0], tip_length=0.3, scale=arrow_length)
        z_arrow = pv.Arrow(start=origin, direction=[0, 0, 1], tip_length=0.3, scale=arrow_length)
        self.plotter.add_mesh(x_arrow, color='red', name='X-axis')
        self.plotter.add_mesh(y_arrow, color='green', name='Y-axis')
        self.plotter.add_mesh(z_arrow, color='blue', name='Z-axis')

    def init_ui(self):
        layout = QVBoxLayout()

        self.sliders = {}
        slider_specs = {
            "rx": [-180, 180, 0],
            "ry": [-180, 180, 0],
            "rz": [-180, 180, 0],
            "tx": [-100, 100, 0],
            "ty": [-100, 100, 0],
            "tz": [-100, 100, 0],
            "scale": [50, 200, 100],
        }

        for name, (min_val, max_val, default) in slider_specs.items():
            hbox = QHBoxLayout()
            label = QLabel(f"{name.upper()}: {default}")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(default)
            slider.valueChanged.connect(lambda val, n=name, l=label: l.setText(f"{n.upper()}: {val}"))
            slider.sliderReleased.connect(self.update_plot)
            hbox.addWidget(label)
            hbox.addWidget(slider)
            layout.addLayout(hbox)
            self.sliders[name] = slider

        fiducial_btn = QPushButton("Pick 3 Surface Fiducials")
        fiducial_btn.clicked.connect(self.pick_surface_fiducials)
        layout.addWidget(fiducial_btn)

        save_btn = QPushButton("Save Transformed Coordinates")
        save_btn.clicked.connect(self.save_transformed_coordinates)
        layout.addWidget(save_btn)

        self.setLayout(layout)

    def get_params(self):
        return {
            "rx": self.sliders["rx"].value(),
            "ry": self.sliders["ry"].value(),
            "rz": self.sliders["rz"].value(),
            "tx": self.sliders["tx"].value() / 1000.0,
            "ty": self.sliders["ty"].value() / 1000.0,
            "tz": self.sliders["tz"].value() / 1000.0,
            "scale": self.sliders["scale"].value() / 100.0,
        }

    def update_plot(self):
        params = self.get_params()
        transformed = self.original_coords * params['scale']
        rot = R.from_euler('zyx', [params['rz'], params['ry'], params['rx']], degrees=True)
        transformed = rot.apply(transformed)
        transformed += np.array([params['tx'], params['ty'], params['tz']])
        self.transformed_coords = transformed

        if self.glyph_actor:
            self.plotter.remove_actor(self.glyph_actor)

        points = pv.PolyData(transformed)
        sphere = pv.Sphere(radius=0.005)
        glyphs = points.glyph(geom=sphere, scale=False)
        self.glyph_actor = self.plotter.add_mesh(glyphs, color='red')
        self.plotter.render()

    def pick_surface_fiducials(self):
        self.surface_fiducials = []
        print("Select 3 fiducials on the surface...")

        def callback(point):
            self.surface_fiducials.append(point)
            if len(self.surface_fiducials) == 3:
                self.align_using_fiducials()

        self.plotter.enable_point_picking(callback=callback, use_picker=True, show_message=True, show_point=True)

    def align_using_fiducials(self):
        src = self.fiducial_coords  # from electrode file
        dst = np.array(self.surface_fiducials)  # clicked

        src_mean = src.mean(0)
        dst_mean = dst.mean(0)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean

        U, S, Vt = np.linalg.svd(np.dot(dst_centered.T, src_centered))
        Rmat = np.dot(U, Vt)
        if np.linalg.det(Rmat) < 0:
            U[:, -1] *= -1
            Rmat = np.dot(U, Vt)

        t = dst_mean - Rmat @ src_mean

        aligned = (Rmat @ self.original_coords.T).T + t
        self.transformed_coords = aligned

        if self.glyph_actor:
            self.plotter.remove_actor(self.glyph_actor)
        points = pv.PolyData(aligned)
        sphere = pv.Sphere(radius=0.005)
        glyphs = points.glyph(geom=sphere, scale=False)
        self.glyph_actor = self.plotter.add_mesh(glyphs, color='red')
        self.plotter.render()

    def save_transformed_coordinates(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Text Files (*.txt);;All Files (*)")
        if filename:
            out = np.column_stack((remaining_labels, self.transformed_coords))
            np.savetxt(filename, out, fmt="%s\t%.6f\t%.6f\t%.6f")
            print(f"Saved to {filename}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ElectrodeAligner()
    window.resize(600, 500)
    window.show()
    sys.exit(app.exec_())
