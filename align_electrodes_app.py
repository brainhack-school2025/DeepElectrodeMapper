import sys
import os
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import (
    QApplication, QWidget, QSlider, QVBoxLayout, QLabel,
    QHBoxLayout, QPushButton, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from scipy.spatial.transform import Rotation as R


class ElectrodeAligner(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Electrode Alignment Tool")

        self.select_obj_folder()

        self.original_coords = self.remaining_coords.copy()
        self.transformed_coords = self.original_coords.copy()
        self.surface_fiducials = []

        self.plotter = BackgroundPlotter(show=True)
        self.load_mesh()
        self.add_axes()

        self.glyph_actor = None

        self.init_ui()
        self.update_plot()

    def select_obj_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Subject OBJ Folder")
        if not folder:
            QMessageBox.critical(self, "Error", "No folder selected.")
            sys.exit()

        self.obj_folder = folder
        basename = os.path.basename(folder)
        subj_id = basename.split("_")[0]  # e.g., "sub-012"

        self.obj_file = os.path.join(folder, "model_mesh.obj")
        self.tex_file = os.path.join(folder, "model_texture.jpg")
        self.txt_file = os.path.join(folder, f"{subj_id}_aligned_electrodes.txt")

        if not os.path.exists(self.obj_file) or not os.path.exists(self.txt_file):
            QMessageBox.critical(self, "Error", "OBJ or electrode file not found in folder.")
            sys.exit()

        self.labels = []
        self.coords = []

        with open(self.txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 4:
                    continue
                label, x, y, z = parts
                self.coords.append([float(x), float(y), float(z)])
                self.labels.append(label)

        self.coords = np.array(self.coords) #/ 1000.0  # mm → meters
        fid_labels = ['nas', 'lhj', 'rhj']
        self.fiducial_coords = np.array([self.coords[self.labels.index(lab)] for lab in fid_labels])
        self.remaining_labels = [lab for lab in self.labels if lab not in fid_labels]
        self.remaining_coords = np.array([self.coords[i] for i, lab in enumerate(self.labels) if lab not in fid_labels])

    def load_mesh(self):
        self.mesh = pv.read(self.obj_file)
        self.mesh = self.mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
        if os.path.exists(self.tex_file):
            #self.mesh.texture_map_to_plane(inplace=True)
            texture = pv.read_texture(self.tex_file)
            self.plotter.add_mesh(self.mesh, texture=texture)
        else:
            self.plotter.add_mesh(self.mesh, color="lightgray", opacity=0.8)

    def add_axes(self):
        origin = np.array([0, 0, 0])
        arrow_length = 0.05
        self.plotter.add_mesh(pv.Arrow(origin, [1, 0, 0], scale=arrow_length), color='red', name='X-axis')
        self.plotter.add_mesh(pv.Arrow(origin, [0, 1, 0], scale=arrow_length), color='green', name='Y-axis')
        self.plotter.add_mesh(pv.Arrow(origin, [0, 0, 1], scale=arrow_length), color='blue', name='Z-axis')

    def init_ui(self):
        layout = QVBoxLayout()
        self.sliders = {}
        specs = {
            "rx": [-180, 180, 0],
            "ry": [-180, 180, 0],
            "rz": [-180, 180, 0],
            "tx": [-100, 100, 0],
            "ty": [-100, 100, 0],
            "tz": [-100, 100, 0],
            "scale": [50, 200, 100],
        }

        for name, (min_val, max_val, default) in specs.items():
            hbox = QHBoxLayout()
            label = QLabel(f"{name.upper()}: {default}")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(default)
            slider.valueChanged.connect(lambda val, n=name, l=label: l.setText(f"{n.upper()}: {val}"))
            slider.sliderReleased.connect(self.update_plot)
            hbox.addWidget(label)
            hbox.addWidget(slider)
            layout.addLayout(hbox)
            self.sliders[name] = slider

        pick_btn = QPushButton("Pick 3 Surface Fiducials")
        pick_btn.clicked.connect(self.pick_surface_fiducials)
        layout.addWidget(pick_btn)

        save_btn = QPushButton("Save Transformed Coordinates")
        save_btn.clicked.connect(self.save_transformed_coordinates)
        layout.addWidget(save_btn)

        self.setLayout(layout)

    def get_params(self):
        return {
            k: (self.sliders[k].value() / (1000.0 if k in ["tx", "ty", "tz"] else 100.0 if k == "scale" else 1.0))
            for k in self.sliders
        }

    def update_plot(self):
        params = self.get_params()
        coords = self.original_coords * params["scale"]

        # Step 1: calculate rotation center (centroid of fiducials)
        center = self.fiducial_coords.mean(axis=0)

        # Step 2: shift to center, rotate, shift back
        shifted = coords - center
        rot = R.from_euler('zyx', [params["rz"], params["ry"], params["rx"]], degrees=True)
        rotated = rot.apply(shifted)
        rotated += center

        # Step 3: apply translation
        translated = rotated + np.array([params["tx"], params["ty"], params["tz"]])
        self.transformed_coords = translated

        if self.glyph_actor:
            self.plotter.remove_actor(self.glyph_actor)

        points = pv.PolyData(self.transformed_coords)
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
        src = self.fiducial_coords
        dst = np.array(self.surface_fiducials)

        src_mean = src.mean(0)
        dst_mean = dst.mean(0)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean

        U, _, Vt = np.linalg.svd(dst_centered.T @ src_centered)
        Rmat = U @ Vt
        if np.linalg.det(Rmat) < 0:
            U[:, -1] *= -1
            Rmat = U @ Vt

        T = dst_mean - Rmat @ src_mean
        aligned = (Rmat @ self.original_coords.T).T + T
        self.transformed_coords = aligned

        if self.glyph_actor:
            self.plotter.remove_actor(self.glyph_actor)
        pts = pv.PolyData(aligned)
        glyph = pts.glyph(geom=pv.Sphere(radius=0.005), scale=False)
        self.glyph_actor = self.plotter.add_mesh(glyph, color='red')
        self.plotter.render()

    def save_transformed_coordinates(self):
        out_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Text Files (*.txt);;All Files (*)")
        if out_path:
            with open(out_path, 'w') as f:
                # Write transformed remaining electrodes
                for label, coord in zip(self.remaining_labels, self.transformed_coords):
                    f.write(f"{label}\t{coord[0]:.6f}\t{coord[1]:.6f}\t{coord[2]:.6f}\n")
                # Write fiducials first (unaltered, in meters)
                fid_labels = ['nas', 'lhj', 'rhj']
                for label, coord in zip(fid_labels, self.fiducial_coords):
                    f.write(f"{label}\t{coord[0]:.6f}\t{coord[1]:.6f}\t{coord[2]:.6f}\n")

            print(f"Saved to {out_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ElectrodeAligner()
    window.resize(600, 500)
    window.show()
    sys.exit(app.exec_())
