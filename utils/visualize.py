import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from textwrap import wrap
import torch
from pytorch3d.transforms import quaternion_to_matrix

OBJECT_KINEMATICS = {
    "largetable": [(0, 2), (1, 3), (2, 3), (2, 6), (3, 7), (4, 6), (5, 7), (6, 7)],
    "clothesstand": [(0, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
    "smallbox": [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)],
}
def get_object_vertices(pos, rot, object="largetable"):
    T = pos.shape[0]
    R = rot
    if object == "largetable":
        hu, hl, w = 0.09, 0.36, 0.28
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
    elif object == "clothesstand":
        offsets = torch.tensor([
            [ 0.75748584, -0.9570913 , -0.20249086],    # top
            [-0.31622088,  0.40539299,  0.08523018],    # bottom center
            [-0.41252733,  0.30123311,  0.2233453],     # bottom square
            [-0.44069129,  0.33733911, -0.05288494],    # bottom square
            [-0.19175047,  0.47344687,  0.2233453],     # bottom square
            [-0.21991443,  0.50955287, -0.05288494],    # bottom square
        ], dtype=torch.float32).unsqueeze(0).expand(T, -1, -1)  # (T, 8, 3)
    elif object == "smallbox":
        offsets = torch.tensor([
            [-0.20109183,  0.09395005,  0.02345553],
            [-0.18586666,  0.05281884, -0.1272752 ],
            [ 0.00053567,  0.22285539,  0.0086462 ],
            [ 0.01576084,  0.18172418, -0.14208453],
            [-0.01115937, -0.19196568,  0.12066076],
            [ 0.0040658 , -0.23309689, -0.03006997],
            [ 0.19046813, -0.06306034,  0.10585143],
            [ 0.2056933 , -0.10419156, -0.0448793 ],
        ], dtype=torch.float32).unsqueeze(0).expand(T, -1, -1)  # (T, 8, 3)
    return torch.bmm(offsets, R.transpose(1, 2)) + pos.unsqueeze(1)

def plot_3d_points(save_path, kinematic_tree, joints, object_name, obj_points, title, dataset, figsize=(3, 3), fps=120, radius=3, show_joints=True):
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
    if object_name is not None and obj_points is not None and len(obj_points) > 0:
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
            for pair in OBJECT_KINEMATICS[object_name]:
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
