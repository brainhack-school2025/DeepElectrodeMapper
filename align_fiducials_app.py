import numpy as np
import os
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import QPushButton, QApplication, QHBoxLayout, QWidget
import sys

# === Parameters ===
subj = "sub-012"
base_dir = "sourcedata"
subj_dir = os.path.join(base_dir, f"{subj}_obj")

obj_file = os.path.join(subj_dir, "model_mesh.obj")
txt_file = os.path.join(base_dir, f"{subj}_ses-01_acq-structure_electrodes.txt")
output_file = os.path.join(subj_dir, f"{subj}_aligned_electrodes.txt")
texture_file = os.path.join(subj_dir, "model_texture.jpg")

# === Load electrode file ===
def load_electrodes(txt_file):
    coords = {}
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                label, x, y, z = parts
                coords[label] = np.array([float(x), float(y), float(z)]) / 1000.0
    return coords

# === Try flips and apply rigid Kabsch alignment ===
def try_flips_and_align_kabsch(elec_points, fidu_points):
    best_err = float('inf')
    best_R = None
    best_t = None
    best_flip = None

    flips = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
        (-1, -1, 1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1)
    ]

    for fx, fy, fz in flips:
        flipped = elec_points * np.array([fx, fy, fz])
        C1 = flipped.mean(axis=0)
        C2 = fidu_points.mean(axis=0)
        A = flipped - C1
        B = fidu_points - C2

        H = A.T @ B
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = C2 - R @ C1
        aligned = (R @ flipped.T).T + t
        err = np.linalg.norm(aligned - fidu_points)

        if err < best_err:
            best_err = err
            best_R = R
            best_t = t
            best_flip = (fx, fy, fz)

    return best_R, best_t, best_flip

# === Perform rigid alignment ===
def align_to_picked_fiducials(electrodes, picked_points, output_file):
    fidu_labels = ['nas', 'lhj', 'rhj']
    elec_points = np.array([electrodes[label] for label in fidu_labels])
    fidu_points = np.array(picked_points)

    # Use Kabsch algorithm with best flip
    R, t, flip = try_flips_and_align_kabsch(elec_points, fidu_points)
    print(f"✅ Best axis flip: {flip}")

    # Apply transform to all electrodes
    aligned_electrodes = {}
    for label, coord in electrodes.items():
        coord_flipped = coord * np.array(flip)
        aligned = (R @ coord_flipped) + t
        aligned_electrodes[label] = aligned

    # Save to file
    with open(output_file, 'w') as f:
        for label, coord in aligned_electrodes.items():
            f.write(f"{label} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

    return aligned_electrodes

# === Launch GUI ===
def run_alignment_gui(obj_file, electrodes, output_file, texture_file=None):
    mesh = pv.read(obj_file)
    mesh = mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)

    plotter = BackgroundPlotter()

    # Create the hover marker (yellow translucent sphere)
    hover_marker = pv.Sphere(radius=0.002)
    hover_actor = plotter.add_mesh(hover_marker, color='yellow', opacity=0.5)
    hover_actor.SetVisibility(False)


    if texture_file and os.path.exists(texture_file):
        texture = pv.read_texture(texture_file)
        plotter.add_mesh(mesh, texture=texture)
    else:
        plotter.add_mesh(mesh, color="lightgray", opacity=0.5)

    fiducial_labels = ['nas', 'lhj', 'rhj']
    picked_points = []
    arrows = []

    msg = plotter.add_text(f"Pick: {fiducial_labels[len(picked_points)]}", font_size=10)

    def update_message():
        if len(picked_points) < 3:
            msg.SetText(0, f"Pick: {fiducial_labels[len(picked_points)]}")
        else:
            msg.SetText(0, "Picked all. Click 'Done'.")

    def hover_callback(caller, event):
        if len(picked_points) >= 3:
            hover_actor.SetVisibility(False)
            return

        # Get mouse position from caller (VTK object)
        x, y = caller.GetEventPosition()
        picked = plotter.pick_mouse_position(x, y)
        if picked is None:
            hover_actor.SetVisibility(False)
            return

        point_id = mesh.find_closest_point(picked)
        picked_point = mesh.points[point_id]
        normal = mesh.point_normals[point_id]

        camera_pos = np.array(plotter.camera_position[0])
        to_camera = camera_pos - picked_point
        to_camera /= np.linalg.norm(to_camera)

        if np.dot(to_camera, normal) < 0:
            hover_actor.SetVisibility(False)
            return

        hover_marker.center = picked_point
        hover_actor.SetVisibility(True)
        plotter.render()

    def pick_callback(point):
        if len(picked_points) < 3:
            point_id = mesh.find_closest_point(point)
            picked_point = mesh.points[point_id]
            normal = mesh.point_normals[point_id]

            picked_points.append(picked_point)

            # Draw a small red sphere at the picked point
            sphere = pv.Sphere(radius=0.003, center=picked_point)
            actor = plotter.add_mesh(sphere, color='red')
            arrows.append(actor)  # reuse arrows list to track for removal

            if len(picked_points) == 3:
                plotter.disable_picking()

            update_message()


    plotter.iren.add_observer("MouseMoveEvent", hover_callback)


    # Always disable before enabling
    plotter.disable_picking()
    plotter.enable_point_picking(
        callback=pick_callback,
        pick_type="point",
        show_message=False,
        show_point=False
    )


    def done_alignment():
        if len(picked_points) != 3:
            print("❌ You must pick exactly 3 points (nas, lhj, rhj).")
            return
        aligned = align_to_picked_fiducials(electrodes, picked_points, output_file)
        print(f"✅ Saved aligned electrodes to {output_file}")
        aligned_coords = np.array(list(aligned.values()))
        plotter.add_points(aligned_coords, color='red', point_size=10, render_points_as_spheres=True)
        msg.SetText(0, "✅ Aligned electrodes plotted.")
        plotter.render()

    def undo_last_pick():
        if picked_points:
            picked_points.pop()
            last_actor = arrows.pop()
            plotter.remove_actor(last_actor)
            plotter.disable_picking()
            plotter.enable_point_picking(
                callback=pick_callback,
                pick_type="point",
                show_message=False,
                show_point=False
            )
            update_message()
            plotter.render()


    # Add Done and Back buttons
    done_btn = QPushButton("Done")
    done_btn.clicked.connect(done_alignment)

    back_btn = QPushButton("Back")
    back_btn.clicked.connect(undo_last_pick)

    # Create a horizontal layout for the buttons
    button_widget = QWidget()
    button_layout = QHBoxLayout()
    button_layout.addWidget(back_btn)
    button_layout.addWidget(done_btn)
    button_widget.setLayout(button_layout)

    # Add the horizontal layout to the main layout
    main_layout = plotter.app_window.centralWidget().layout()
    main_layout.addWidget(button_widget)

# === Main run ===
if __name__ == "__main__":
    electrodes = load_electrodes(txt_file)
    run_alignment_gui(obj_file, electrodes, output_file, texture_file=texture_file)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.exec_()
