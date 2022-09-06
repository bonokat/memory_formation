from __future__ import annotations
import mne
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


def create_annotations(lines: list[str]) -> list[mne.Annotations]: # create annotaitons from a list of strings
    processed_lines = list() # create an empty list

    for line in lines: # go line by line in txt
        splitted_lines = line.split(sep=', ') # split the lines in 3 parts (name, onset, duration)
        splitted_lines[1] = float(splitted_lines[1]) # convert onset into float
        splitted_lines[2] = float(splitted_lines[2]) # convert duration into float
        processed_lines.append(splitted_lines) # append new list with "BAD", float(onset), float(duration)
    
    processed_lines = np.array(processed_lines) # convert new list to np.array
    
    return mne.Annotations(         # return new list in mne.Annotations format
        processed_lines[:, 1],
        processed_lines[:, 2],
        processed_lines[:, 0]
    )


def annotate_raw(raw: mne.io.Raw, events: Optional[np.ndarray] = None) -> mne.io.Raw: # annotate raw files
    events = events.astype(float)
    if events is not None: # if there are events, we add them to the raw file plot
        fig = raw.plot(events=events) 
    else:                  # if not, we plot raw file 
        fig = raw.plot()
    plt.show()

    fig.canvas.key_press_event('a') # now we can add events manually: 1. don't close the small window; 
                                    # 2. press 'a' to add annotations; 3. first close small window, then close plot

    return raw  #return annotated raw file


def create_events(event_lines: list[str]) -> list[tuple[np.array, dict]]:
    
    event_names = list() # empty list for event names
    line_contents = list() # empry list to put lines

    for line in event_lines: 
        line_content = line.split(', ') # split lines into name of event, time, zero
        event_names.append(line_content[0]) # list the event names
        line_contents.append(line_content) # list all info about events

    unique_events = set(event_names) # sets contain only unique objects
    events_ids = dict() # empty dictionary to store info on events

    for id, event in enumerate(unique_events): # assing numbers to unique event names
        events_ids[event] = id # ????

    events = list()

    for line_content in line_contents:
        events.append([line_content[1], 0, events_ids[line_content[0]]]) # events format for mne: time, 0, name)

    events = np.array(events) #convert list to np.array

    return events, events_ids
