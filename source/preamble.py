import os
from enum import Enum

### FLAGS
KAGGLE = False

VOTE_COLS = ['seizure_vote',
 'lpd_vote',
 'gpd_vote',
 'lrda_vote',
 'grda_vote',
 'other_vote']

COLOR_MAP = {
    "Seizure":"r",
    "GPD":"g",
    "GRDA":"b",
    "LRDA":"pink",
    "LPD":"orange",
    "Other":"grey"
}

### CONST
class Const(int, Enum):
    eeg_len = 50
    fq_eeg = 200

class Grade(float, Enum):
    certain = 1.
    # TO COMPLETE



### FLAG DEPENDANT
if KAGGLE:
    base_dir = os.path.join(os.getcwd(), "..", "input", "hms-harmful-brain-activity-classification")
    
    class Dir(str, Enum):
        root = base_dir
        eeg_train = os.path.join(base_dir, "train_eegs")
        eeg_test = os.path.join(base_dir, "test_eegs")
        spc_train = os.path.join(base_dir, "train_spectrograms")
        spc_test = os.path.join(base_dir, "test_spectrograms")
        out = os.getcwd()

else: # local
    base_dir = os.path.join(os.getcwd(), "..")
    
    RANDOM_STATE = 1

    class Dir(str, Enum):
        root = base_dir
        eeg_train = os.path.join(base_dir, "train_eegs")
        eeg_test = os.path.join(base_dir, "test_eegs")
        spc_train = os.path.join(base_dir, "train_spectrograms")
        spc_test = os.path.join(base_dir, "test_spectrograms")
        out = os.path.join(base_dir, "submissions")

    class Const(int, Enum):
        eeg_len = 50
        fq_eeg = 200

    class Grade(float, Enum):
        certain = 1.
        # TO COMPLETE
