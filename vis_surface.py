import os
import torch
import numpy as np
from os.path import exists, join
import h5py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
from svg import utils
import hydra

norm = 'filter'
ignore = 'biasbn'
xmin = -0.003
ymin = -0.003
xmax = 0.003
ymax = 0.003
xnum = 71
ynum = 71
zmin = -300
zmax = 0
#ckpt_file = 'exp/local/2022.07.17/1729_test/latest.pth'
#ckpt_file = 'exp/walker2d/2022.07.29/0623_horizon10_sn_s0/best.pth'

#@hydra.main(config_path='config', config_name='train')
#def vis_surface(cfg):
   # env = utils.make_norm_env(cfg)
   # cfg.obs_dim = int(env.observation_space.shape[0])
   # cfg.action_dim = env.action_space.shape[0]
   # cfg.action_range = [
   #     float(env.action_space.low.min()),
   #     float(env.action_space.high.max())
   # ]
 #   agent = hydra.utils.instantiate(cfg.agent)
 #   agent.load_checkpoint(ckpt_path=ckpt_pth, evaluate=True)
def vis_surface(agent, env, step, device):
    save_dir = "/content/drive/MyDrive/RPgrad/vis_surface"
    policy_weights = [p.data for p in agent.actor.parameters()]
   # for w in policy_weights:
   #     print(w)
    random_x_direction = [torch.randn(w.size()).to(device) for w in policy_weights]
    random_y_direction = [torch.randn(w.size()).to(device) for w in policy_weights]

    normalize_directions_for_weights(random_x_direction, policy_weights, norm, ignore)
    normalize_directions_for_weights(random_y_direction, policy_weights, norm, ignore)

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
  #  del_list = os.listdir(save_dir)
   # for f in del_list:
   #     file_path = os.path.join(save_dir, f)
   #     if os.path.isfile(file_path):
   #         os.remove(file_path)
    dir_file = join(save_dir, 'dir.h5')
    surf_file = join(save_dir, 'surface.h5')
    loss_curve_file = join(save_dir, 'loss_curve.h5')
    if os.path.isfile(dir_file):
        os.remove(dir_file)
    if os.path.isfile(surf_file):
        os.remove(surf_file)
    if os.path.isfile(loss_curve_file):
        os.remove(loss_curve_file)
    assert not exists(dir_file)
    f = h5py.File(dir_file, 'w')
    write_list(f, 'xdirection', random_x_direction)
    write_list(f, 'ydirection', random_y_direction)
    f.close()
    print("direction file created: %s" % dir_file)
    d = [random_x_direction, random_y_direction]
    setup_surface_file(surf_file, dir_file)

    f = h5py.File(surf_file, 'r+')
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    loss_key = 'train_loss'
    acc_key = 'train_acc'
    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates), len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        f[loss_key] = losses
        f[acc_key] = accuracies
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]
    coor_list = []
    x_loss = []
    y_loss = []
    fig = plt.figure()
    for x_coor, y_coor in zip(xcoordinates, ycoordinates):
        coor_list.append(x_coor)
        set_weights(agent.actor, policy_weights, [random_x_direction], x_coor)
        obs = torch.Tensor(env.reset()).unsqueeze(0).to(device)
        exp_rew, _, _ = agent.expand_Q(obs, agent.critic, sample=True, discount=False)
        exp_rew = (exp_rew / agent.horizon).item()
        x_loss.append(-exp_rew)

        set_weights(agent.actor, policy_weights, [random_x_direction], y_coor)
        exp_rew, _, _ = agent.expand_Q(obs, agent.critic, sample=True, discount=False)
        exp_rew = (exp_rew / agent.horizon).item()
        y_loss.append(-exp_rew)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(coor_list, x_loss)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(coor_list, y_loss)

    # Save the full figure...
    fig.savefig(join(save_dir, str(step) + '_loss.png'))
    txt_file_name = join(save_dir, str(step) + '_loss.txt')
    with open(txt_file_name, 'w') as txt_f:
        txt_f.write(str([coor_list, x_loss, y_loss]))


    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = get_job_indices(losses, xcoordinates, ycoordinates, None)
    # Loop over all uncalculated loss values
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]
        # Load the weights corresponding to those coordinates into the net
        set_weights(agent.actor, policy_weights, d, coord)

        avg_reward = []
        for episode in range(1):
            obs = torch.Tensor(env.reset()).unsqueeze(0).to(device)
            exp_rew, _, _ = agent.expand_Q(obs, agent.critic, sample=True, discount=False)
            exp_rew = exp_rew / 15
            avg_reward.append(exp_rew.detach())
        avg_reward = torch.mean(torch.cat(avg_reward))
#        print(avg_reward, "est_reward")
   #     print(evaluate(agent, env), "eval_real")

        acc = avg_reward
        loss = -avg_reward
        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc
        f[loss_key][:] = losses
        f[acc_key][:] = accuracies
        f.flush()
        print('%d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f' % (
                count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
                acc_key, acc))
    f.close()
    plot_2d_contour(step, surf_file, 'train_loss', zmin, zmax, 0.5, False)


def plot_2d_contour(step, surf_file, surf_name='train_loss', vmin=0.1, vmax=10, vlevel=0.5, show=False):
    """Plot 2D contour map and 3D surface."""

    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    elif surf_name == 'train_err' or surf_name == 'test_err' :
        Z = 100 - np.array(f[surf_name][:])
    else:
        print ('%s is not found in %s' % (surf_name, surf_file))

    print('------------------------------------------------------------------')
    print('plot_2d_contour')
    print('------------------------------------------------------------------')
    print("loading surface file: " + surf_file)
    print('len(xcoordinates): %d   len(ycoordinates): %d' % (len(x), len(y)))
    print('max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(Z), surf_name, np.min(Z)))
    print(Z)

    if (len(x) <= 1 or len(y) <= 1):
        print('The length of coordinates is not enough for plotting contours')
        return

    """
    # --------------------------------------------------------------------
    # Plot 2D contours
    # --------------------------------------------------------------------
    fig = plt.figure()
    CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    plt.clabel(CS, inline=1, fontsize=8)
    fig.savefig(surf_file + str(step) + '_' + surf_name + '_2dcontour' + '.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    fig = plt.figure()
    print(surf_file + str(step) + '_' + surf_name + '_2dcontourf' + '.pdf')
    CS = plt.contourf(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    fig.savefig(surf_file + str(step) + '_' + surf_name + '_2dcontourf' + '.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    # --------------------------------------------------------------------
    # Plot 2D heatmaps
    # --------------------------------------------------------------------
    fig = plt.figure()
    sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
                           xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(surf_file + str(step) + '_' + surf_name + '_2dheat.pdf',
                                  dpi=300, bbox_inches='tight', format='pdf')
    """
    # --------------------------------------------------------------------
    # Plot 3D surface
    # --------------------------------------------------------------------
   # fig = plt.figure()
   # ax = Axes3D(fig)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_zlim(-0.01, 2.01)
    ax._axis3don = False
    surf = ax.plot_surface(X, Y, Z - np.min(Z), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(surf_file + str(step) + '_' + surf_name + '_3dsurface.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    f.close()
    if show:
        plt.show()


def set_weights(net, weights, directions=None, step=None):
    """
        Overwrite the network's weights with a specified list of tensors
        or change weights along directions with a step size.
    """
    if directions is None:
        # You cannot specify a step length without a direction.
        for (p, w) in zip(net.parameters(), weights):
            p.data.copy_(w.type(type(p.data)))
    else:
        assert step is not None, 'If a direction is specified then step must be specified as well'

        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d*step for d in directions[0]]

        for (p, w, d) in zip(net.parameters(), weights, changes):
            p.data = w + torch.Tensor(d).type(type(w)).to('cuda')


def evaluate(agent, env):
    episode_rewards = []
    for episode in range(5):
        obs = env.reset()
        agent.reset()
        done = False
        episode_reward = 0
        count = 0
        while not done:
            count += 1
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=False)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)
    return np.mean(episode_rewards)



def normalize_direction(direction, weights, norm='filter'):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.
        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm()/direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())


def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
    """
        The normalization scales the direction entries according to the entries of weights.
    """
    assert(len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0) # ignore directions for weights with 1 dimension
            else:
                d.copy_(w) # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)


def setup_surface_file(surf_file, dir_file):
    # skip if the direction file already exists
    if os.path.exists(surf_file):
        print ("%s is already set up" % surf_file)
        return

    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file

    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(xmin, xmax, num=xnum)
    f['xcoordinates'] = xcoordinates

    ycoordinates = np.linspace(ymin, ymax, num=ynum)
    f['ycoordinates'] = ycoordinates
    f.close()

    return surf_file

def get_unplotted_indices(vals, xcoordinates, ycoordinates=None):
    """
    Args:
      vals: values at (x, y), with value -1 when the value is not yet calculated.
      xcoordinates: x locations, i.e.,[-1, -0.5, 0, 0.5, 1]
      ycoordinates: y locations, i.e.,[-1, -0.5, 0, 0.5, 1]
    Returns:
      - a list of indices into vals for points that have not yet been calculated.
      - a list of corresponding coordinates, with one x/y coordinate per row.
    """

    # Create a list of indices into the vectorizes vals
    inds = np.array(range(vals.size))

    # Select the indices of the un-recorded entries, assuming un-recorded entries
    # will be smaller than zero. In case some vals (other than loss values) are
    # negative and those indexces will be selected again and calcualted over and over.
    inds = inds[vals.ravel() <= 0]

    # Make lists containing the x- and y-coodinates of the points to be plotted
    if ycoordinates is not None:
        # If the plot is 2D, then use meshgrid to enumerate all coordinates in the 2D mesh
        xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
        s1 = xcoord_mesh.ravel()[inds]
        s2 = ycoord_mesh.ravel()[inds]
        return inds, np.c_[s1,s2]
    else:
        return inds, xcoordinates.ravel()[inds]


def split_inds(num_inds, nproc):
    """
    Evenly slice out a set of jobs that are handled by each MPI process.
      - Assuming each job takes the same amount of time.
      - Each process handles an (approx) equal size slice of jobs.
      - If the number of processes is larger than rows to divide up, then some
        high-rank processes will receive an empty slice rows, e.g., there will be
        3, 2, 2, 2 jobs assigned to rank0, rank1, rank2, rank3 given 9 jobs with 4
        MPI processes.
    """

    chunk = num_inds // nproc
    remainder = num_inds % nproc
    splitted_idx = []
    for rank in range(0, nproc):
        # Set the starting index for this slice
        start_idx = rank * chunk + min(rank, remainder)
        # The stopping index can't go beyond the end of the array
        stop_idx = start_idx + chunk + (rank < remainder)
        splitted_idx.append(range(start_idx, stop_idx))

    return splitted_idx


def get_job_indices(vals, xcoordinates, ycoordinates, comm):
    """
    Prepare the job indices over which coordinate to calculate.
    Args:
        vals: the value matrix
        xcoordinates: x locations, i.e.,[-1, -0.5, 0, 0.5, 1]
        ycoordinates: y locations, i.e.,[-1, -0.5, 0, 0.5, 1]
        comm: MPI environment
    Returns:
        inds: indices that splitted for current rank
        coords: coordinates for current rank
        inds_nums: max number of indices for all ranks
    """

    inds, coords = get_unplotted_indices(vals, xcoordinates, ycoordinates)

    rank = 0 if comm is None else comm.Get_rank()
    nproc = 1 if comm is None else comm.Get_size()
    splitted_idx = split_inds(len(inds), nproc)

    # Split the indices over the available MPI processes
    inds = inds[splitted_idx[rank]]
    coords = coords[splitted_idx[rank]]

    # Figure out the number of jobs that each MPI process needs to calculate.
    inds_nums = [len(idx) for idx in splitted_idx]

    return inds, coords, inds_nums


def write_list(f, name, direction):
    """ Save the direction to the hdf5 file with name as the key
        Args:
            f: h5py file object
            name: key name_surface_file
            direction: a list of tensors
    """

    grp = f.create_group(name)
    for i, l in enumerate(direction):
        if isinstance(l, torch.Tensor):
            l = l.cpu().numpy()
        grp.create_dataset(str(i), data=l)


if __name__ == "__main__":
    work_dir = os.getcwd()
  #  ckpt_pth = join(work_dir, ckpt_file)
    save_dir = join(work_dir, 'vis_surface/')
    vis_surface()