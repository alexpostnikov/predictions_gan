import numpy as np
from matplotlib import pyplot as plt, cm
import uuid


def fig_canvas_image(fig):
    """Returns a [H, W, 3] uint8 np.array image from fig.canvas.tostring_rgb()."""
    # Just enough margin in the figure to display xticks and yticks.
    fig.subplots_adjust(
        left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))



def get_colormap(num_agents):
    """Compute a color map array of shape [num_agents, 4]."""
    colors = cm.get_cmap('jet', num_agents)
    colors = colors(range(num_agents))
    np.random.shuffle(colors)
    return colors


def get_viewport(all_states, all_states_mask):
    """Gets the region containing the data.

  Args:
    all_states: states of agents as an array of shape [num_agents, num_steps,
      2].
    all_states_mask: binary mask of shape [num_agents, num_steps] for
      `all_states`.

  Returns:
    center_y: float. y coordinate for center of data.
    center_x: float. x coordinate for center of data.
    width: float. Width of data.
  """
    valid_states = all_states[all_states_mask]
    all_y = valid_states[..., 1]
    all_x = valid_states[..., 0]

    center_y = (np.max(all_y) + np.min(all_y)) / 2
    center_x = (np.max(all_x) + np.min(all_x)) / 2

    range_y = np.ptp(all_y)
    range_x = np.ptp(all_x)

    width = max(range_y, range_x)

    return center_y, center_x, width

def prepare_data_for_vis(decoded_example):
    past_states = np.stack(
        [decoded_example['state/past/x'], decoded_example['state/past/y']],
        -1).reshape(-1, 10, 2)
    past_states_mask = decoded_example['state/past/valid'].reshape(128, 10) > 0.0
    # [num_agents, 1, 2] float32.
    current_states = np.stack(
        [decoded_example['state/current/x'], decoded_example['state/current/y']],
        -1).reshape(-1, 1, 2)
    current_states_mask = decoded_example['state/current/valid'].reshape(128, 1) > 0.0
    # [num_agents, num_future_steps, 2] float32.
    future_states = np.stack(
        [decoded_example['state/future/x'], decoded_example['state/future/y']],
        -1).reshape(-1, 80, 2)
    future_states_mask = decoded_example['state/future/valid'].reshape(128, 80) > 0.0
    # [num_points, 3] float32.
    roadgraph_xyz = decoded_example['roadgraph_samples/xyz'].reshape(-1, 3)
    num_agents, num_past_steps, _ = past_states.shape
    num_future_steps = future_states.shape[1]
    color_map = get_colormap(num_agents)
    # [num_agens, num_past_steps + 1 + num_future_steps, depth] float32.
    all_states = np.concatenate([past_states, current_states, future_states], 1)
    # [num_agens, num_past_steps + 1 + num_future_steps] float32.
    all_states_mask = np.concatenate(
        [past_states_mask, current_states_mask, future_states_mask], 1)
    center_y, center_x, width = get_viewport(all_states, all_states_mask)
    return center_x, center_y, color_map, current_states, current_states_mask, future_states, future_states_mask, \
           num_future_steps, num_past_steps, past_states, past_states_mask, roadgraph_xyz, width


def create_figure_and_axes(size_pixels):
    """Initializes a unique figure and axes for plotting."""
    fig, ax = plt.subplots(1, 1, num=uuid.uuid4())

    # Sets output image to pixel resolution.
    dpi = 100
    size_inches = size_pixels / dpi
    fig.set_size_inches([size_inches, size_inches])
    fig.set_dpi(dpi)
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='y', colors='black')
    fig.set_tight_layout(True)
    ax.grid(False)
    return fig, ax


def vis_cur_and_fut(decoded_example, predictions=None, size_pixels=1000, bn=0, confs=None):
    data = {}
    for key in decoded_example.keys():
        try:
            data[key] = decoded_example[key][bn].detach().cpu().numpy()
        except:
            pass
    center_x, center_y, color_map, current_states, current_states_mask, future_states, future_states_mask, \
    num_future_steps, num_past_steps, past_states, past_states_mask, roadgraph_xyz, width = prepare_data_for_vis(
        data)
    images = []
    s = current_states
    m = current_states_mask
    prediction = None
    if predictions is not None:
        prediction = predictions.clone() #[(data["state/tracks_to_predict"]>0).nonzero()].detach().cpu().numpy()
        # prediction = prediction[:,0] #+ current_states[:,np.newaxis][data["state/tracks_to_predict"]>0]
        # # predictions = predictions.cumsum(1)

    future_states_mask *= np.repeat(data["state/tracks_to_predict"].reshape(128, 1), 80, axis=1)>0
    im = visualize_one_step_with_future(s[:, 0], m[:, 0], future_states, future_states_mask, roadgraph_xyz,
                            'cur with fut', center_y, center_x, width, color_map, size_pixels,
                                        predictions=prediction, confs=confs)
    return im


def visualize_one_step_with_future(states, mask, future_states, future_states_mask, roadgraph, title,
                                   center_y, center_x, width, color_map, size_pixels=1000, predictions=None, confs=None):
    """Generate visualization for a single step."""

    # Create figure and axes.
    fig, ax = create_figure_and_axes(size_pixels=size_pixels)

    # Plot roadgraph.
    rg_pts = roadgraph[:, :2].T
    ax.plot(rg_pts[0, :], rg_pts[1, :], 'k.', alpha=1, ms=2)

    masked_x = states[:, 0][mask]
    masked_y = states[:, 1][mask]
    colors = color_map[mask]

    # Plot agent current position.
    ax.scatter(
        masked_x,
        masked_y,
        marker='o',
        linewidths=2,
        color=colors,
    )
    for ped in range(128):

        maskeds_x = []
        maskeds_y = []
        for step in range(future_states.shape[1]):
            if not future_states_mask[ped,step]:
                continue
            masked_x = future_states[ped, step, 0] #[future_states_mask[:,step]]
            masked_y = future_states[ped, step, 1] #[future_states_mask[:,step]]
            maskeds_x.append(masked_x)
            maskeds_y.append(masked_y)
        colors = color_map[ped] #+ np.array([0.3,0.3,0.3,0.3])
        ax.plot( maskeds_x, maskeds_y, linewidth=4, color=colors)
        ax.plot( maskeds_x, maskeds_y, linewidth=2, color=np.array([181, 179, 92, 255])/255.)

    nump, timestamps, modalities, datadim = predictions.shape
    # nump = confs.shape[0]
    if predictions is not None:
        for ped in range(nump):
            # if future_states_mask[ped].sum() == 0:
            #     continue
            for modality in range(modalities):
                maskeds_x = []
                maskeds_y = []
                for step in range(timestamps):
                    # if not future_states_mask[ped, step]:
                    #     continue
                    # if [future_states_mask[ped, step]]:
                        masked_x = predictions[ped, step, modality, 0]
                        masked_y = predictions[ped, step, modality, 1]
                        maskeds_x.append(masked_x)
                        maskeds_y.append(masked_y)
                        colors = color_map[ped]
                    # ax.scatter(
                    #     masked_x,
                    #     masked_y,
                    #     marker='o',
                    #     linewidths=0.05,
                    #     color=colors,
                    # )
                if confs is None:
                    conf = 0.3
                if confs is not None:
                    conf = confs[ped, modality].detach().cpu().item()
                ax.plot(
                    maskeds_x,
                    maskeds_y,
                    # marker='o',
                    linewidth=2*conf,
                    linestyle='--',
                    color=colors - 0.3*np.array([0, 0, 0, 1-conf]),
                )
                ax.text(maskeds_x[-1], maskeds_y[-1], f"{conf:.2f}",
                        fontsize="xx-small")

    # Title.
    ax.set_title(title)

    # Set axes.  Should be at least 10m on a side and cover 160% of agents.
    size = max(10, width * 1.0)
    ax.axis([
        -size / 2 + center_x, size / 2 + center_x, -size / 2 + center_y,
        size / 2 + center_y
    ])
    ax.set_aspect('equal')

    image = fig_canvas_image(fig)

    plt.close(fig)

    return image

