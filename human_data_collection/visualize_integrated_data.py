#!/usr/bin/env python
"""3D visualization tool for integrated_recordings HDF5 data."""

import argparse, os, sys  # noqa: E401
import h5py
import numpy as np
import plotly.graph_objects as go

HAND_KEYPOINTS_SEMANTIC = {
    0: "wrist", 1: "thumb_mcp", 2: "thumb_pip", 3: "thumb_dip", 4: "thumb_tip",
    5: "index_mcp", 6: "index_pip", 7: "index_dip", 8: "index_inter", 9: "index_tip",
    10: "middle_mcp", 11: "middle_pip", 12: "middle_dip", 13: "middle_inter", 14: "middle_tip",
    15: "ring_mcp", 16: "ring_pip", 17: "ring_dip", 18: "ring_inter", 19: "ring_tip",
    20: "pinky_mcp", 21: "pinky_pip", 22: "pinky_dip", 23: "pinky_inter", 24: "pinky_tip",
}

HAND_CONNECTIONS = {
    "thumb":  [(0, 1), (1, 2), (2, 3), (3, 4)],
    "index":  [(0, 5), (5, 6), (6, 7), (7, 8), (8, 9)],
    "middle": [(0, 10), (10, 11), (11, 12), (12, 13), (13, 14)],
    "ring":   [(0, 15), (15, 16), (16, 17), (17, 18), (18, 19)],
    "pinky":  [(0, 20), (20, 21), (21, 22), (22, 23), (23, 24)],
}

FINGER_COLORS = {
    "thumb": "green", "index": "blue", "middle": "red",
    "ring": "purple", "pinky": "yellow",
}

FINGERTIP_INDICES = [4, 9, 14, 19, 24]


def _transform_keypoints_to_world(keypoints: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Transform hand keypoints (25,3) from local to world coords via a 4x4 matrix."""
    homo = np.ones((keypoints.shape[0], 4))
    homo[:, :3] = keypoints
    return (matrix @ homo.T).T[:, :3]


def batch_transform_hand_keypoints(batch: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    """Batch version: (N,25,3) keypoints + (N,4,4) matrices -> (N,25,3) world coords."""
    result = np.zeros_like(batch)
    for i in range(batch.shape[0]):
        result[i] = _transform_keypoints_to_world(batch[i], transforms[i])
    return result


def _extract_axes(matrix: np.ndarray, scale: float) -> tuple:
    """Extract origin and axis endpoints from a 4x4 transform matrix."""
    rot, pos = matrix[:3, :3], matrix[:3, 3]
    return pos, pos + rot[:, 0] * scale, pos + rot[:, 1] * scale, pos + rot[:, 2] * scale


def batch_extract_coordinate_axes(batch: np.ndarray, scale: float = 0.1):
    """Batch version: (N,4,4) matrices -> (positions, x_ends, y_ends, z_ends) each (N,3)."""
    n = batch.shape[0]
    positions = np.zeros((n, 3))
    x_axes = np.zeros((n, 3))
    y_axes = np.zeros((n, 3))
    z_axes = np.zeros((n, 3))
    for i in range(n):
        positions[i], x_axes[i], y_axes[i], z_axes[i] = _extract_axes(batch[i], scale)
    return positions, x_axes, y_axes, z_axes


def create_coordinate_frame_trace(position, x_end, y_end, z_end, name_prefix, show_legend=True):
    """Create Plotly traces for a coordinate frame (origin + 3 axis lines)."""
    return [
        go.Scatter3d(x=[position[0]], y=[position[1]], z=[position[2]],
                     mode='markers', marker=dict(size=6, color='black'),
                     name=f'{name_prefix}_center', showlegend=show_legend),
        go.Scatter3d(x=[position[0], x_end[0]], y=[position[1], x_end[1]], z=[position[2], x_end[2]],
                     mode='lines', line=dict(color='red', width=4),
                     name=f'{name_prefix}_X', showlegend=show_legend),
        go.Scatter3d(x=[position[0], y_end[0]], y=[position[1], y_end[1]], z=[position[2], y_end[2]],
                     mode='lines', line=dict(color='green', width=4),
                     name=f'{name_prefix}_Y', showlegend=show_legend),
        go.Scatter3d(x=[position[0], z_end[0]], y=[position[1], z_end[1]], z=[position[2], z_end[2]],
                     mode='lines', line=dict(color='blue', width=4),
                     name=f'{name_prefix}_Z', showlegend=show_legend),
    ]


def create_hand_traces(hand_keypoints_world, hand_side, show_legend=True):
    """Create Plotly traces for hand keypoints, finger connections, and thumb-index distance."""
    traces = []
    main_color = 'lightgreen' if hand_side == 'left' else 'lightcoral'

    thumb_tip = hand_keypoints_world[4]
    index_tip = hand_keypoints_world[9]
    thumb_index_dist = float(np.linalg.norm(thumb_tip - index_tip))
    distance_cm = thumb_index_dist * 100

    # Keypoints
    traces.append(go.Scatter3d(
        x=hand_keypoints_world[:, 0], y=hand_keypoints_world[:, 1], z=hand_keypoints_world[:, 2],
        mode='markers', marker=dict(size=3, color=main_color, opacity=0.7),
        name=f'{hand_side}_keypoints', showlegend=show_legend,
        text=[f'{hand_side}_{HAND_KEYPOINTS_SEMANTIC.get(i, f"point_{i}")}' for i in range(25)],
        hovertemplate='%{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>',
    ))

    # Finger connections
    for finger_name, connections in HAND_CONNECTIONS.items():
        for s, e in connections:
            traces.append(go.Scatter3d(
                x=[hand_keypoints_world[s, 0], hand_keypoints_world[e, 0]],
                y=[hand_keypoints_world[s, 1], hand_keypoints_world[e, 1]],
                z=[hand_keypoints_world[s, 2], hand_keypoints_world[e, 2]],
                mode='lines', line=dict(color=FINGER_COLORS[finger_name], width=2),
                name=f'{hand_side}_{finger_name}', showlegend=False,
            ))

    # Fingertips
    fingertips = hand_keypoints_world[FINGERTIP_INDICES]
    traces.append(go.Scatter3d(
        x=fingertips[:, 0], y=fingertips[:, 1], z=fingertips[:, 2],
        mode='markers',
        marker=dict(size=5, color='darkred' if hand_side == 'right' else 'darkgreen', symbol='diamond'),
        name=f'{hand_side}_fingertips', showlegend=show_legend,
    ))

    # Thumb-index distance line (color-coded by proximity)
    if distance_cm < 2:
        line_color, line_width = 'red', 6
    elif distance_cm < 5:
        line_color, line_width = 'orange', 5
    elif distance_cm < 10:
        line_color, line_width = 'yellow', 4
    else:
        line_color, line_width = 'blue', 3

    traces.append(go.Scatter3d(
        x=[thumb_tip[0], index_tip[0]], y=[thumb_tip[1], index_tip[1]], z=[thumb_tip[2], index_tip[2]],
        mode='lines+text', line=dict(color=line_color, width=line_width),
        name=f'{hand_side}_thumb_index_distance', showlegend=show_legend,
        text=['', f'{distance_cm:.1f}cm'], textposition='middle center',
        textfont=dict(size=12, color=line_color),
        hovertemplate=f'{hand_side} thumb-index distance: {distance_cm:.2f}cm<extra></extra>',
    ))

    return traces, thumb_index_dist


def load_integrated_data(file_path: str) -> dict:
    """Load hand/head pose data from an integrated_recordings HDF5 file."""
    data = {}
    with h5py.File(file_path, 'r') as f:
        print(f"Loading: {file_path}")
        data['head_matrices'] = f['raw/head_mat'][:]
        data['left_wrist_matrices'] = f['raw/left_wrist_mat'][:]
        data['right_wrist_matrices'] = f['raw/right_wrist_mat'][:]
        data['left_hand_keypoints'] = f['raw/left_keypoints'][:]
        data['right_hand_keypoints'] = f['raw/right_keypoints'][:]
        if 'observation/timestamp' in f:
            data['timestamps'] = f['observation/timestamp'][:]
        data['n_frames'] = data['head_matrices'].shape[0]
        print(f"  {data['n_frames']} frames, hands: {data['left_hand_keypoints'].shape}")
    return data


def visualize_integrated_data(data: dict, output_file: str | None = None):
    """Create an animated 3D visualization of the integrated recording data."""
    print("Transforming keypoints & extracting axes...")
    left_hand_world = batch_transform_hand_keypoints(data['left_hand_keypoints'], data['left_wrist_matrices'])
    right_hand_world = batch_transform_hand_keypoints(data['right_hand_keypoints'], data['right_wrist_matrices'])
    lw_pos, lw_x, lw_y, lw_z = batch_extract_coordinate_axes(data['left_wrist_matrices'], scale=0.1)
    rw_pos, rw_x, rw_y, rw_z = batch_extract_coordinate_axes(data['right_wrist_matrices'], scale=0.1)

    fig = go.Figure()
    axis_len = 0.3

    # World origin and axes
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers',
                               marker=dict(size=10, color='black'), name='World origin'))
    for ax, color, label in [([axis_len, 0, 0], 'red', 'X'), ([0, axis_len, 0], 'green', 'Y'),
                              ([0, 0, axis_len], 'blue', 'Z')]:
        fig.add_trace(go.Scatter3d(x=[0, ax[0]], y=[0, ax[1]], z=[0, ax[2]], mode='lines',
                                   line=dict(color=color, width=6), name=f'World {label}'))

    # Initial frame traces
    for t in create_coordinate_frame_trace(lw_pos[0], lw_x[0], lw_y[0], lw_z[0], 'left_wrist'):
        fig.add_trace(t)
    for t in create_coordinate_frame_trace(rw_pos[0], rw_x[0], rw_y[0], rw_z[0], 'right_wrist'):
        fig.add_trace(t)

    lt, ld = create_hand_traces(left_hand_world[0], 'left')
    for t in lt:
        fig.add_trace(t)
    rt, rd = create_hand_traces(right_hand_world[0], 'right')
    for t in rt:
        fig.add_trace(t)

    print(f"Frame 0 — left thumb-index: {ld*100:.2f}cm, right: {rd*100:.2f}cm")
    print("Building animation frames...")
    frames, left_dists, right_dists = [], [], []
    world_origin = [
        go.Scatter3d(x=[0], y=[0], z=[0], mode='markers',
                     marker=dict(size=10, color='black'), showlegend=False),
        go.Scatter3d(x=[0, axis_len], y=[0, 0], z=[0, 0], mode='lines',
                     line=dict(color='red', width=6), showlegend=False),
        go.Scatter3d(x=[0, 0], y=[0, axis_len], z=[0, 0], mode='lines',
                     line=dict(color='green', width=6), showlegend=False),
        go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, axis_len], mode='lines',
                     line=dict(color='blue', width=6), showlegend=False),
    ]
    for i in range(data['n_frames']):
        ft = list(world_origin)
        ft.extend(create_coordinate_frame_trace(lw_pos[i], lw_x[i], lw_y[i], lw_z[i], 'left_wrist', False))
        ft.extend(create_coordinate_frame_trace(rw_pos[i], rw_x[i], rw_y[i], rw_z[i], 'right_wrist', False))
        lht, ldist = create_hand_traces(left_hand_world[i], 'left', False)
        ft.extend(lht)
        left_dists.append(ldist)
        rht, rdist = create_hand_traces(right_hand_world[i], 'right', False)
        ft.extend(rht)
        right_dists.append(rdist)
        frames.append(go.Frame(data=ft, name=str(i)))
    fig.frames = frames

    # Distance statistics
    left_dists_cm = np.array(left_dists) * 100
    right_dists_cm = np.array(right_dists) * 100
    print("\n=== Thumb-index distance statistics ===")
    for label, d in [("Left", left_dists_cm), ("Right", right_dists_cm)]:
        print(f"{label} hand ({len(d)} frames): min={np.min(d):.2f}cm  max={np.max(d):.2f}cm  "
              f"mean={np.mean(d):.2f}cm  std={np.std(d):.2f}cm")
        grasp = np.sum(d < 3.0)
        if grasp:
            print(f"  Potential grasp frames (< 3cm): {grasp}")

    # Layout
    fig.update_layout(
        title="Integrated Recordings 3D Visualization",
        showlegend=True,
        scene=dict(
            aspectmode='data', aspectratio=dict(x=1, y=1, z=1),
            camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0),
                        eye=dict(x=-1.5, y=0, z=0.5)),
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
        ),
        updatemenus=[dict(
            type='buttons', direction='left', showactive=True, x=0.1, y=0.1,
            buttons=[
                dict(label='Play', method='animate',
                     args=[None, dict(frame=dict(duration=100, redraw=True),
                                      fromcurrent=True, mode='immediate',
                                      transition=dict(duration=50))]),
                dict(label='Pause', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=True),
                                        mode='immediate', transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            currentvalue=dict(prefix='Frame: '), pad=dict(t=50),
            len=0.9, x=0.1, xanchor='left', y=0, yanchor='top',
            steps=[dict(args=[[str(i)], dict(frame=dict(duration=100, redraw=True),
                                             mode='immediate', transition=dict(duration=50))],
                        label=str(i), method='animate')
                   for i in range(data['n_frames'])],
        )],
    )

    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        fig.write_html(output_file)
        print(f"Saved: {output_file}")
    else:
        fig.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Integrated recordings 3D visualization tool')
    parser.add_argument('--file', '-f',
                        default='data/integrated_recordings/test/episode_0.hdf5',
                        type=str, help='Path to HDF5 file')
    parser.add_argument('--output', '-o', default=None,
                        type=str, help='Output HTML file path (optional)')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: file not found: {args.file}")
        sys.exit(1)

    print(f"Processing file: {args.file}")
    data = load_integrated_data(args.file)
    visualize_integrated_data(data, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
