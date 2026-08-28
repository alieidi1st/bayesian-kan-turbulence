"""
Shared publication-quality plotting style for the BKAN study.

A single place to fix figure typography, line weights, resolution and colour-bar
formatting so every figure in the paper/report is visually consistent and
journal-ready (vector PDF, embedded fonts, properly sized and labelled colour
bars). Import and call `set_pub_style()` at the top of any figure script; use
`add_colorbar()` for fields so the bar height always matches its axes.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Column widths (inches) for a typical two-column journal (e.g. JCP / Elsevier).
COL_SINGLE = 3.5
COL_ONEHALF = 5.5
COL_DOUBLE = 7.2


def set_pub_style():
    """Apply a clean, consistent rcParams profile for publication figures."""
    mpl.rcParams.update({
        'figure.dpi': 150,            # on-screen; savefig overrides for export
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,           # embed TrueType (editable text, no Type-3)
        'ps.fonttype': 42,
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Computer Modern Roman'],
        'mathtext.fontset': 'dejavuserif',
        'font.size': 9,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.4,
        'lines.markersize': 4,
        'axes.grid': True,
        'grid.linewidth': 0.4,
        'grid.alpha': 0.35,
        'legend.frameon': False,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })


def add_colorbar(fig, mappable, ax, label='', size='4%', pad=0.06,
                 orientation='vertical'):
    """
    Attach a colour bar whose extent matches the host axes height (vertical) or
    width (horizontal), with a labelled axis. Returns the Colorbar.
    """
    divider = make_axes_locatable(ax)
    side = 'right' if orientation == 'vertical' else 'bottom'
    cax = divider.append_axes(side, size=size, pad=pad)
    cb = fig.colorbar(mappable, cax=cax, orientation=orientation)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.6)
    return cb


def save(fig, path_noext, formats=('pdf', 'png')):
    """Save a figure to multiple formats (vector PDF + raster PNG) and close it."""
    for fmt in formats:
        fig.savefig(f'{path_noext}.{fmt}')
    plt.close(fig)
