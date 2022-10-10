from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as m
import mne


def plot_tfr(
    power: mne.time_frequency.AverageTFR,
    cmap: Optional[str | m.colors.Colormap] = 'RdBu_r',
) -> m.figure.Figure:

    fig, ax = plt.subplots(1,1)
    ax.imshow(
        power.data.mean(0), aspect='auto', origin='lower', cmap=cmap,
        extent=(power.times[0], power.times[-1], power.freqs[0], power.freqs[-1])
    )
    ax.set_yticks(np.linspace(power.freqs[0], power.freqs[-1], power.freqs.shape[0]))
    ax.set_yticklabels(np.round(power.freqs, 2))

    return fig
