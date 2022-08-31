from __future__ import annotations
import mne
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


def create_annotations(lines: list[str]) -> list[mne.Annotations]:
    processed_lines = list()

    for line in lines:
        splitted_lines = line.split(sep=', ')
        splitted_lines[1] = float(splitted_lines[1])
        splitted_lines[2] = float(splitted_lines[2])
        processed_lines.append(splitted_lines)
    
    processed_lines = np.array(processed_lines)
    
    return mne.Annotations(
        processed_lines[:, 1],
        processed_lines[:, 2],
        processed_lines[:, 0]
    )


def annotate_raw(raw: mne.io.Raw, events: Optional[np.ndarray] = None) -> mne.io.Raw:
    if events is not None:
        fig = raw.plot(events=events) 
    else:
        fig = raw.plot()
    plt.show()

    fig.canvas.key_press_event('a')

    return raw
