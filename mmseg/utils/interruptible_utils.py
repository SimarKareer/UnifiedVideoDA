import os
import os.path as osp
import signal
import threading

import torch

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", 0)
STATE_FILE = osp.join(
    os.environ["HOME"], ".interrupted_states", "{}.pt".format(SLURM_JOB_ID)
)

REQUEUE = threading.Event()
REQUEUE.clear()
EXIT = threading.Event()
EXIT.clear()


def _requeue_handler(signum, frame):
    # define the handler function
    # note that this is not executed here, but rather
    # when the associated signal is sent
    print("signaled for requeue")
    EXIT.set()
    REQUEUE.set()


def _clean_exit_handler(signum, frame):
    EXIT.set()
    print("Exiting cleanly", flush=True)


def init_handlers():
    signal.signal(signal.SIGUSR1, _requeue_handler)

    # signal.signal(signal.SIGINT, _clean_exit_handler)
    signal.signal(signal.SIGTERM, _clean_exit_handler)
    signal.signal(signal.SIGUSR2, _clean_exit_handler)


def save_state(state_dict):
    torch.save(state_dict, STATE_FILE)

def requeue():
    print("requeuing job " + os.environ["SLURM_JOB_ID"])
    os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])

def save_and_requeue(state_dict):
    torch.save(state_dict, STATE_FILE)
    print("requeuing job " + os.environ["SLURM_JOB_ID"])
    os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])


def get_requeue_state():
    if osp.exists(STATE_FILE):
        return torch.load(STATE_FILE, map_location="cpu")
    else:
        return None