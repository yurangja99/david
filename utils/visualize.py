import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from textwrap import wrap
import torch
from pytorch3d.transforms import quaternion_to_matrix


def get_object_vertices(pos, rot, hu=0.09, hl=0.36, w=0.28, h=0.1):
    T = pos.shape[0]
    R = rot
    offsets = torch.tensor([
        [-w, -hl, -w],
        [-w, -hl,  w],
        [-w,  hu, -w],
        [-w,  hu,  w],
        [ w, -hl, -w],
        [ w, -hl,  w],
        [ w,  hu, -w],
        [ w,  hu,  w],
    ], dtype=torch.float32).unsqueeze(0).expand(T, -1, -1)  # (T, 8, 3)
    return torch.bmm(offsets, R.transpose(1, 2)) + pos.unsqueeze(1)

def plot_3d_points(save_path, kinematic_tree, joints, obj_points, title, dataset, figsize=(3, 3), fps=120, radius=3, show_joints=True):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    data = joints.copy().reshape(len(joints), -1, 3)
    obj_data = None
    if obj_points is not None and len(obj_points) > 0:
        obj_data = obj_points.copy().reshape(len(obj_points), -1, 3)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
        if obj_data is not None:
            obj_data *= 0.003
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
        if obj_data is not None:
            obj_data *= 1.3
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization
        if obj_data is not None:
            obj_data *= -1.5

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = fig.add_subplot(111, projection='3d')
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    
    frame_number = data.shape[0]
    trajec = data[:, 0, [0, 2]]
    if obj_data is not None:
        obj_data[..., 0] -= data[:, 0:1, 0]
        obj_data[..., 2] -= data[:, 0:1, 2]
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        ax.clear()

        ax.view_init(elev=120, azim=-90)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        # ax.set_title(title, fontsize=10)
        ax.grid(False)
        plt.axis('off')

        plot_xzPlane(
            MINS[0] - trajec[index, 0],
            MAXS[0] - trajec[index, 0],
            0,
            MINS[2] - trajec[index, 1],
            MAXS[2] - trajec[index, 1]
        )
        if show_joints:
            ax.scatter(data[index, :, 0], data[index, :, 1], data[index, :, 2], color='black', s=3, alpha=0.2)
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(
                data[index, chain, 0],
                data[index, chain, 1],
                data[index, chain, 2],
                linewidth=linewidth,
                color=color
            )
        if obj_data is not None:
            ax.scatter(obj_data[index, :, 0], obj_data[index, :, 1], obj_data[index, :, 2], color=colors_blue[0], s=3, alpha=0.2)
        if obj_data is not None and obj_data.shape[1] == 8:
            for pair in [(0, 2), (1, 3), (2, 3),
                         (2, 6), (3, 7), (4, 6), (5, 7), (6, 7)]:
                linewidth = 2.0
                ax.plot3D(
                    obj_data[index, pair, 0],
                    obj_data[index, pair, 1],
                    obj_data[index, pair, 2],
                    linewidth=linewidth,
                    color=colors_blue[0],
                )

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()
