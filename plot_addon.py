#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some functions and classes for graphic representation.

@author: misiak
"""

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import os

from stats_addon import cdf_calc

def lighten_color(color, amount=0.5):
    """
    Credits to: 
    Ian Hincks ihincks
    https://gist.github.com/ihincks/6a420b599f43fcd7dbd79d56798c4e5a
    
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    
    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0],1-amount * (1-c[1]),c[2])


# subtitles in legend
import matplotlib.text as mtext
class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, r'\underline{' + orig_handle + '}',
                           usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title


def custom_autoscale(axis, xdata, ydata):
    
    gold2 = ((1+5**0.5)/2)**0.5
    
    xmin, xmax = np.min(xdata), np.max(xdata)
    ymin, ymax = np.min(ydata), np.max(ydata)
    
    xcore = (xmax + xmin)/2
    ycore = (ymax + ymin)/2
    
    xside = (xmax - xmin) * gold2
    yside = (ymax - ymin) * gold2
    
    xinf, xsup = xcore - xside/2, xcore + xside/2
    yinf, ysup = ycore - yside/2, ycore + yside/2
    
    axis.set_xlim(xinf, xsup)
    axis.set_ylim(yinf, ysup)


def ax_hist(axis, bin_edges, data_array, lab, color='slateblue'):
    """ Draw pretty histogramm and cdf in given axis.
    Return bin_array, hist_array, cdf_array.    
    """
    c_dark = lighten_color(color, 1.5)
    c_light = lighten_color(color, 0.8)
    
    style = [pe.Normal(), pe.withStroke(foreground='k', linewidth=3)]

    bin_array = bin_edges[1:] - (bin_edges[1]-bin_edges[0])/2

    data_hist, _ = np.histogram(data_array, bins=bin_edges)
    
    data_sorted, cdf_array = cdf_calc(data_array)
    
    hist_line, = axis.plot(bin_array, data_hist, drawstyle='steps-mid',
                           color=c_dark)

    axis.fill_between(bin_array, data_hist, label=lab,
                      color=c_light, step='mid')

    a0 = axis.twinx()
    a0.set_ylabel('CDF', color='grey')
    a0.tick_params(axis='y', labelcolor='grey')
    
    cdf_line, = a0.plot(data_sorted, cdf_array,
                        drawstyle='steps', color=color, path_effects=style)
    
    axis.grid(True)
    axis.set_ylabel('Counts Events {}'.format(lab), color='k')
    axis.tick_params(axis='y', labelcolor='k')
    axis.set_xlabel('Energy [ADU]')
    
    axis.legend(loc=2)
    
    axis.set_yscale('log')
    axis.set_xlim(bin_edges[0], bin_edges[-1])
    
    return a0, bin_array, data_hist, cdf_array



def ax_hist_v2(axis, bin_edges, data_array, lab=None, color='slateblue', weights=None):
    """ Draw pretty histogramm and cdf in given axis.
    Return ax, bin_array, hist_array, cdf_array.    
    """
    import matplotlib.patheffects as pe
    
    c_dark = lighten_color(color, 1.5)
    c_light = lighten_color(color, 0.8)
    
    style = [pe.Normal(), pe.withStroke(foreground='k', linewidth='3')]

    bin_array = bin_edges[1:] - (bin_edges[1]-bin_edges[0])/2

    data_hist, _ = np.histogram(data_array, bins=bin_edges)
    
    data_sorted, cdf_array = cdf_calc(data_array, weights=weights)
    
    data_sorted = np.insert(data_sorted, 0, -np.inf)
    cdf_array = np.insert(cdf_array, 0, 0)
    
    # hist_line, = axis.plot(bin_array, data_hist, drawstyle='steps',
    #                         color=c_dark)

    # axis.fill_between(bin_array, data_hist, label=lab,
    #                   color=c_light, step='pre')

    a0 = axis.twinx()

    a0.set_ylabel('CDF', color='k')
    a0.tick_params(axis='y', labelcolor='dimgrey')
    
    cdf_line, = a0.plot(data_sorted, cdf_array,
                        drawstyle='steps',
                        color='k',
                        # path_effects=style,
    )
    
    # axis.grid(True)
    # axis.set_ylabel('Counts Events {}'.format(lab), color='k')
    # axis.tick_params(axis='y', labelcolor='k')
    # axis.set_xlabel('Energy [ADU]')
    
    # axis.legend(loc=2)
    
    # axis.set_yscale('log')
    axis.set_xlim(bin_edges[0], bin_edges[-1])
    
    a0.set_ylim(0, 1.1)
    
    return a0, bin_array, data_hist, cdf_array



def plot_ion_vs_ion(ana, energy_array, **kwargs):
    """
    Quick and dirty for run61.
    """

    # general
    run_info = ' '.join([ana.run, ana.detector])
    run_tree = ana.all.run_tree    
    
    # recovering data
#    energy = trig.filt_decor.Energy_OF
    energy = energy_array
    
    # initializing pseudo-corner plot
    ax_tuples = [(0,0), (1,0), (1,1), (2,0), (2,1), (2,2)]
    ax_discard = [(0, 1), (1, 2), (0, 2)]
    
    chan_x = np.insert(run_tree.chan_veto, 0, run_tree.chan_collect[1])
    chan_y = np.append(run_tree.chan_veto, run_tree.chan_collect[0])
    
    num = '{} : Ion vs Ion CUSTOM'.format(run_info)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8.27, 8.27),
                             num=num, sharex='col', sharey='row')
    
    
    options = {'ls':'none', 'marker':'.', 'zorder':9, 'color':'k',}
    options.update(kwargs)
    
    # actually plotting the data
    for atupl in ax_tuples:
        
        ax = axes[atupl]
        xind = chan_x[atupl[1]]
        yind = chan_y[atupl[0]]
    
        energy_x = energy[:, xind]
        energy_y = energy[:, yind]
            
        ax.plot(
                energy_x, energy_y,
                label='10kev events',
                **options
        )
    

        custom_autoscale(ax, energy_x, energy_y)
        
        ax.grid(alpha=0.3)
        
        if atupl == (0,0):
            ax.legend(loc='lower left', framealpha=1,
                      bbox_to_anchor=(1.05, 0.05), borderaxespad=0.,
            )
        
        if atupl[0] == 2:
            ax.set_xlabel(
                    'Energy {} [ADU]'.format(
                            run_tree.chan_label[xind].replace('_', ' ')
                    )
            )
                
        if atupl[1] == 0:
            ax.set_ylabel(
                    'Energy {} [ADU]'.format(
                            run_tree.chan_label[yind].replace('_', ' ')
                    )
            )
    
    fig.text(0.65, 0.98, num,
             horizontalalignment='center',
             verticalalignment='center',
             bbox=dict(facecolor='lime', alpha=0.5))
    
    for tupl in ax_discard:
        fig.delaxes(axes[tupl])
    fig.tight_layout()
    fig.subplots_adjust(hspace=.0, wspace=.0)
    
    return fig


def basic_corner(samples, labels, axes=None, num='Basic corner plot', **kwargs):
    
    assert samples.ndim == 2
    
    clen, cnum = samples.shape
    assert clen > cnum
    
    nplot = cnum - 1
    
    new_flag = False
    if axes is None:
        fig, axes = plt.subplots(nrows=nplot, ncols=nplot, figsize=(9, 9),
                                 num=num, sharex='col', sharey='row')
        new_flag = True
    
    else:
        fig = axes.flatten()[0].get_figure()
#    # lower triangle axes without diagonal
#    axes_active = np.tril(axes, -1)
#    # upper triangle axes with diagonal
#    axes_discarded = np.triu(axes, 0)
    
    options = {'ls':'none',
               'marker':'.',
               'zorder':9,
               'color':'k',
               'markersize':2}
    options.update(kwargs)
    
    for i in range(nplot):
        for j in range(nplot):
            ax = axes[i,j]
            
            # removing upper plots
            if i < j:
                try:
                    fig.delaxes(ax)
                except:
                    pass
        
            x_data = samples[:, (cnum-1+j)%cnum]
            y_data = samples[:, i]
            
            ax.plot(
                    x_data, y_data,
                    **options
            )            
            
            custom_autoscale(ax, x_data, y_data)
        
            ax.grid(alpha=0.3)
        
            if (i==0) and (j==0):
                ax.legend(loc='lower left', framealpha=1,
                          bbox_to_anchor=(1.05, 0.05), borderaxespad=0.,
                )
        
            if new_flag:
                if (i==nplot-1):
                    ax.set_xlabel(
                            '{}'.format(
                                    labels[(cnum-1+j)%cnum].replace('_', ' ')
                            )
                    )
                        
                if (j==0):
                    ax.set_ylabel(
                            '{}'.format(
                                    labels[i].replace('_', ' ')
                            )
                    )
    
    if new_flag:
        fig.text(0.65, 0.98, num,
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox=dict(facecolor='lime', alpha=0.5))
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=.0, wspace=.0)
    
    return fig, axes


def save_figure_dict(fig_dict, output_dir, extension='png'):
    os.makedirs(output_dir, exist_ok=True)
    for key, fig in fig_dict.items():
        fig.savefig( output_dir + '/{}.{}'.format(key, extension) )
        