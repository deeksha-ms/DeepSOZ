import os
import mne
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from scipy.spatial import ConvexHull


def make_images(fns, preds, labels, viz_folder, prefix, suffix, threshold=None):
    for fn, pred, label in zip(fns, preds, labels):
        pic_fn = os.path.join(viz_folder,
                              '{}{}{}.png'.format(prefix, fn, suffix))
        plot_yhat(pred, label, fn=pic_fn, threshold=threshold)


def plot_yhat(y_hat, label, fn=None, threshold=None):
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(y_hat[:, 1], label='Prediction')
    ax.plot(label, label='Ground Truth')
    if threshold:
        plt.axhline(threshold, 0, 1, linestyle='--',
                    color='k', label='Threshold')
        classifications = np.asarray(y_hat[:, 1] >= threshold, dtype=int)
        plt.fill_between(np.arange(len(classifications)),
                         y_hat[:, 1] * classifications, 0,
                         color='blue',       # The outline color
                         alpha=0.2)

    ax.set_xlabel('Time (s)', fontsize=16)
    ax.set_ylabel('Model Output', fontsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    # ax.legend(ncol=3, loc='upper right')
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([0, len(y_hat[:, 1])])
    if fn:
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()

    else:
        plt.show()


def plot_curves(xs, ys, labels, xlabel, ylabel, fn=None,
                labelsize=16, legendsize=16, legend=True):
    """Plot curves"""
    for x, y, label in zip(xs, ys, labels):
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel, fontsize=labelsize)
    plt.ylabel(ylabel, fontsize=labelsize)
    if legend:
        plt.legend(fontsize=legendsize)
    if fn:
        plt.savefig(fn)
        plt.close()
    else:
        plt.show()


def plot_attention_output(y_hat, y_hat_chn, attn, labels, channels,
                          fn="", title="", channel_labels=None):
    """Create plots of all the beliefs"""

    # Create 3 plots
    fig, ax = plt.subplots(2, 1)
    ax1 = ax[0]
    ax2 = ax[1]
    if title:
        ax1.set_title(title, pad=40)

    # Plot the metachain in the first state
    ax1.plot(attn, label='Attention')
    ax1.plot(y_hat[:, 1], label='Predicted')
    ax1.plot(y_hat_chn[:, 1], label='Chn Predicted')
    ax1.plot(labels, label='Ground Truth SZ')
    ax1.set_xlim(-0.5, len(attn)+0.5)
    ax1.legend(ncol=4, loc="upper center", bbox_to_anchor=[0.5, 1.25])

    # Show the X beliefs
    ax2.imshow(channels.T, aspect='auto', vmin=0, vmax=1)

    # Set node labels
    if channel_labels:
        nn = len(channel_labels)
        ax2.set_yticks(np.arange(nn))
        ax2.set_yticklabels(channel_labels)

    plt.tight_layout()
    # Save and close
    if fn:
        plt.savefig(fn)
        plt.close()
    else:
        plt.show()


def topoplot(scores, ax,
             label_list= ['FP1-AF1', 'FP2-AF2','F7-AF1','F3-AF1','FZ-AF1','F4-AF2','F8-AF2','T7-AF1','C3-AF1',
             'CZ-AF1', 'C4-AF2','T8-AF2','P7-AF1','P3-AF1','PZ-AF1','P4-AF2','P8-AF2','O1-AF1','O2-AF2'], 
             title=None, fn='',
             plot_hemisphere=False, plot_lobe=False, zone=None,
             lobe_correct=None, lat_correct=None, onset_map=None):
    # Create the layout
    layout = mne.channels.read_layout('EEG1005')
    # positions = []
    pos2d = []
    layout_names = [name.upper() for name in layout.names]
    for ch in label_list:
        if '-' in ch:
            anode, cathode = ch.split('-')
            anode_idx = layout_names.index(anode)
            cathode_idx = layout_names.index(cathode)
            anode_pos = layout.pos[anode_idx, 0:2]
            cathode_pos = layout.pos[cathode_idx, 0:2]
            pos2d.append([(a + c) / 2 for a, c in zip(anode_pos, cathode_pos)])
        else:
            idx = layout_names.index(ch)
            # positions.append(layout.pos[idx, :])
            pos2d.append(layout.pos[idx, 0:2])
    # positions = np.asarray(positions)
    pos2d = np.asarray(pos2d)
    # Scale locations from [-1, 1]    this scaling is wrong!
    pos2d = ([3.3042, 3.5] * pos2d) - ([1.61, 2.1942])


    # fig = plt.figure()
    #ax = plt.gca()
    if onset_map is not None:
        onset_map = np.asarray(onset_map, dtype=bool)
        mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                           linewidth=0, markersize=6)
        im, cn = mne.viz.plot_topomap(scores, pos2d, sphere=1,
                                      axes=ax, vmin=0, vmax=1, show=False,
                                      outlines='head', mask=onset_map,
                                      mask_params=mask_params)

        # Create circles for the points
        radius = 0.2
        points_x = []
        points_y = []
        for dd in np.arange(0, 2*np.pi, 0.1):
            for pp, val in enumerate(onset_map):
                if val:
                    points_x.append(pos2d[pp, 0] + radius*np.cos(dd))
                    points_y.append(pos2d[pp, 1] + radius*np.sin(dd))
        points_x = np.asarray(points_x)
        points_y = np.asarray(points_y)
        circle_points = np.vstack((points_x, points_y)).T
        
        ##convex hull draws a circle/weird sharpe around the true active channels
        hull = ConvexHull(circle_points)
        for simplex in hull.simplices:
            ax.plot(circle_points[simplex, 0], circle_points[simplex, 1], 'k-')

    else:
        im, cn = mne.viz.plot_topomap(scores, pos2d, sphere=1,
                                      axes=ax, vmin=0, vmax=1, show=False,
                                     outlines='head')

    ##############from here all cosmetic   
    #draws vertical and horizontal lines cutting the brain into 4 parts 
    if plot_hemisphere:
        ax.plot([0, 0], [-1, 1], linestyle='--', color='k', linewidth=6)
    if plot_lobe:
        ax.plot([-0.96, 0.96], [0.2, 0.2], linestyle='--', color='k',
                linewidth=6)

    # Thicken outline of main brain circle
    circle = plt.Circle((0, 0), .98, edgecolor='k', facecolor='none',
                        linewidth=6)
    ax.add_artist(circle)
    ax.plot([-0.167, 0, 0.167], [.988, 1.15, .988], color='k', linewidth=6)
    ax.plot([.98, 1.02, 1.05, 1.08, 1.09, 1.06, 1.02, 0.98],
            [.12, .16, .15, .11, -.18, -.26, -.27, -.24], color='k', linewidth=6)
    ax.plot([-.98, -1.02, -1.05, -1.08, -1.09, -1.06, -1.02, -0.98],
            [.12, .16, .15, .11, -.18, -.26, -.27, -.24], color='k', linewidth=6)

    # of the 
    if lobe_correct and lat_correct:
        color = 'green'
    elif lobe_correct or lat_correct:
        color = 'orange'
    elif not lobe_correct and not lat_correct:
        color = 'red'
    else:
        color = 'blue'
    
    #decides the location of colored small circle outside the brain
    if zone == 1:
        circle = plt.Circle((-0.95, 0.8), 0.16, color=color)
        ax.add_artist(circle)
    elif zone == 2:
        circle = plt.Circle((0.95, 0.8), 0.16, color=color)
        ax.add_artist(circle)
    elif zone == 3:
        circle = plt.Circle((-0.95, -0.8), 0.16, color=color)
        ax.add_artist(circle)
    elif zone == 4:
        circle = plt.Circle((0.95, -0.8), 0.16, color=color)
        ax.add_artist(circle)

    if title:
        ax.set_title(title, fontsize=16)
    if fn:
        plt.savefig(fn, bbox_inches='tight')
    #plt.show()



def vis_history(history, fn):
    # nepochs = len()

    # Four axes, returned as a 2-d array
    f, axarr = plt.subplots(3, 3)
    axarr[0, 0].set_title('Total Loss')
    axarr[0, 0].plot(history['train']['total loss'], color='b')
    axarr[0, 0].plot(history['val']['total loss'], color='m')
    axarr[0, 0].set_yscale('log')

    axarr[0, 1].set_title('Total SZ Loss')
    axarr[0, 1].plot(history['train']['total sz loss'], color='b')
    axarr[0, 1].plot(history['val']['total sz loss'], color='m')
    axarr[0, 1].set_yscale('log')

    axarr[0, 2].set_title('Channel SZ Loss')
    axarr[0, 2].plot(history['train']['channel sz loss'], color='b')
    axarr[0, 2].plot(history['val']['channel sz loss'], color='m')
    axarr[0, 2].set_yscale('log')

    axarr[1, 0].set_title('Attention Map Loss Pos')
    axarr[1, 0].plot(history['train']['attn map loss pos'], color='b')
    axarr[1, 0].plot(history['val']['attn map loss pos'], color='m')
    axarr[1, 0].set_yscale('log')

    axarr[1, 1].set_title('Attention Map Loss Neg')
    axarr[1, 1].plot(history['train']['attn map loss neg'], color='b')
    axarr[1, 1].plot(history['val']['attn map loss neg'], color='m')
    axarr[1, 1].set_yscale('log')

    axarr[1, 2].set_title('Attention Map Loss Margin')
    axarr[1, 2].plot(history['train']['attn map loss margin'], color='b')
    axarr[1, 2].plot(history['val']['attn map loss margin'], color='m')
    axarr[1, 2].set_yscale('log')

    axarr[2, 0].set_title('Channel Map Loss Pos')
    axarr[2, 0].plot(history['train']['chn map loss pos'], color='b')
    axarr[2, 0].plot(history['val']['chn map loss pos'], color='m')
    axarr[2, 0].set_yscale('log')

    axarr[2, 1].set_title('Channel Map Loss Neg')
    axarr[2, 1].plot(history['train']['chn map loss neg'], color='b')
    axarr[2, 1].plot(history['val']['chn map loss neg'], color='m')
    axarr[2, 1].set_yscale('log')

    axarr[2, 2].set_title('Channel Map Loss Margin')
    axarr[2, 2].plot(history['train']['chn map loss margin'], color='b')
    axarr[2, 2].plot(history['val']['chn map loss margin'], color='m')
    axarr[2, 2].set_yscale('log')

    plt.tight_layout()
    plt.savefig(fn)
    plt.close()
