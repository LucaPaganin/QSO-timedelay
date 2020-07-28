import logging
import sys
import numpy as np


def configure_logger(logger, logfile):
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    # create stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stdout_handler.setLevel(logging.INFO)
    stderr_handler.setLevel(logging.ERROR)
    # create formatters and add them to the handlers
    logfile_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(logfile_formatter)
    stdout_handler.setFormatter(logging.Formatter('%(message)s'))
    stderr_handler.setFormatter(logging.Formatter('%(message)s'))
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

def get_season_ends(t):
    diffs = np.diff(t)
    season_ends = np.where(diffs >= diffs.mean() + 3*diffs.std())[0]
    season_ends = np.append(season_ends, len(t)-1)
    season_ends.sort()
    
    return season_ends

def get_season_begs(t):
    season_ends = get_season_ends(t)
    season_ends = season_ends[:-1]
    season_begs = season_ends + 1
    season_begs = np.append(season_begs, 0)
    season_begs.sort()
    
    return season_begs
    
def get_season_masks(t):
    diffs = np.diff(t)
    
    season_ends = get_season_ends(t)
    season_begs = get_season_begs(t)
    
    masks = []
    
    for beg, end in zip(season_begs, season_ends):
        masks.append((t >= t[beg]) & (t <= t[end]))
    
    return masks

