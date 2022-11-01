from __future__ import annotations
import mne
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


def create_annotations(lines: list[str], onset: Optional[float] = 0) -> list[mne.Annotations]: # create annotaitons from a list of strings
    processed_lines = list() # create an empty list

    for line in lines: # go line by line in txt
        splitted_lines = line.split(sep=', ') # split the lines in 3 parts (name, onset, duration)
        splitted_lines[1] = float(splitted_lines[1]) # convert onset into float
        splitted_lines[2] = float(splitted_lines[2]) # convert duration into float
        processed_lines.append(splitted_lines) # append new list with "BAD", float(onset), float(duration)

    processed_lines = np.array(processed_lines) # convert new list to np.array

    return mne.Annotations(         # return new list in mne.Annotations format
        processed_lines[:, 1].astype(float) - onset,
        processed_lines[:, 2].astype(float),
        processed_lines[:, 0]
    )

def annotate_raw(raw: mne.io.Raw, events: Optional[np.ndarray] = None) -> mne.io.Raw: # annotate raw files
    if events is not None: # if there are events, we add them to the raw file plot
        fig = raw.plot(events=events)
    else:                  # if not, we plot raw file
        fig = raw.plot()
    plt.show()

    fig.canvas.key_press_event('a') # now we can add events manually: 1. don't close the small window;
                                    # 2. press 'a' to add annotations; 3. first close small window, then close plot

    return raw  #return annotated raw file


def encode_event(event_name: str) -> int:
    encoder = {
        'enc': '1',
        'neg': '1',
        'neu': '0',
        'hits': '1',
        'miss': '0',
        'sure': '1',
        'notsure': '0'
    }
    return int(''.join([encoder.get(event_group, '0') for event_group in event_name.split('/')]))


def create_events(event_lines: list[str],  onset: Optional[float] = 0) -> list[tuple[np.array, dict]]:

    event_names = list() # empty list for event names
    line_contents = list() # empry list to put lines

    for line in event_lines:
        line_content = line.split(', ') # split lines into name of event, time, zero
        line_content[0] = line_content[0].replace('_', '/')
        event_names.append(line_content[0]) # list the event names
        line_contents.append(line_content) # list all info about events

    unique_events = set(event_names) # sets contain only unique objects
    events_ids = dict() # empty dictionary to store info on events

    for event in unique_events: # assing numbers to unique event names
        events_ids[event] = encode_event(event)

    events = list()

    for line_content in line_contents:
        events.append([float(line_content[1]) - onset, 0, events_ids[line_content[0]]]) # events format for mne: time, 0, name)

    events = np.array(events).astype(float) #convert list to np.array

    return events, events_ids


def read_events(raw: mne.io.Raw, path: str, onset: Optional[float] = 0) -> tuple[np.ndarray, dict[str, int]]:
    with open(path) as f:
        lines = f.readlines()

    events, event_ids = create_events(lines, onset)
    events[:, 0] *= raw.info['sfreq']

    return events.astype(int), event_ids

