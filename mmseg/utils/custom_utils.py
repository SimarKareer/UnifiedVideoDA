import os

def three_channel_flow(flow):
    """
    flow: B, C, H, W
    Make C 3 if not already by repeating
    """
    B, C, H, W = flow.shape
    assert H > 100 and W > 100, "Is this actually an image?"
    if C == 1:
        return flow.repeat(1, 3, 1, 1)
    else:
        return flow

def next_available_file(filename, extension):
    """
    filename: absolute path of file to save /a/b/result
    extension: ex) .pkl
    returns: /a/b/resultn.pkl
    """
    MAXITER = 100

    for i in range(MAXITER):
        if not os.path.exists(f"{filename}{i}{extension}"):
            return f"{filename}{i}{extension}"
    
    return None