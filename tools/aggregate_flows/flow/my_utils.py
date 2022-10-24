from tkinter import W
import png
import array
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from benchmark_viper import VIPER
from util_flow import ReadKittiPngFile


def visFlow(flow, image=None, threshold=2.0, skip_amount=30):
    # Don't affect original image
    if image is None:
        H, W, _ = flow.shape
        image = np.zeros((H, W, 3))
    else:
        image = image.copy()

    h, w, _ = flow.shape
    flow_start = np.ones_like(flow)
    flow_start[:, :, 0] = np.ones((h, w)) * np.arange(w)[None, :] #u represenets horizontal mvmnt.  here each number should rep it's x value
    flow_start[:, :, 1] = np.ones((h, w)) * np.arange(h)[:, None]
    flow_end = flow + flow_start
    
    norm = np.linalg.norm(flow_end - flow_start, axis=2)
    norm[norm < threshold] = 0
    
    # Draw all the nonzero values
    nz = np.nonzero(norm)
    # print(nz)
    for i in range(0, len(nz[0]), skip_amount):
        y, x = nz[0][i], nz[1][i]
        
        cv2.arrowedLine(image,
                        pt1=tuple(flow_start[y,x].astype(np.int32)), 
                        pt2=tuple(flow_end[y,x].astype(np.int32)),
                        color=(0, 255, 0), 
                        thickness=1, 
                        tipLength=.2)
    return image

def imshow(img, scale=1):
    """
    img: H, W, 3
    """
    import cv2
    import IPython
    print(img.shape)
    out_dim = np.array(img.shape)
    out_dim = out_dim[out_dim > 3] * scale
    out_dim = out_dim.astype(np.int)
    print(out_dim)
    img = cv2.resize(img, out_dim[[1, 0]])
    # img = img[::2, ::2]
    # img = np.resize(img, out_dim)
    
    _,ret = cv2.imencode('.jpg', img) 
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)

def loadFlow(im_path):
    im = cv2.imread(im_path)
    flow = ReadKittiPngFile(im_path)
    w, h, u, v, mask = ReadKittiPngFile(im_path)
    
    u = np.array(u).reshape((h, w))
    v = np.array(v).reshape((h, w))
    mask = np.array(mask).reshape((h, w))
    # print(np.logical_and((u==v), u!=0).sum())
    # print(np.logical_and((u!=v), u!=0).sum())
    
    flow = np.concatenate((u[None, :, :], v[None, :, :]), axis=0)
    flow[:, np.logical_not(mask.astype(np.bool))] = 0

    return np.transpose(flow, (1, 2, 0))
    

def loadVisFlow(im_path, rgb_im_path=None, skip_amount=500, scale=1):
    flow = loadFlow(im_path)
    
    rgb_im = None if rgb_im_path == None else cv2.imread(rgb_im_path)
    imshow(
        visFlow(flow, rgb_im, skip_amount=skip_amount),
        scale=scale
    )

# def mergeFlow(flow1, flow2):
#     """
#         flow1 / flow2: (H, W, 2)
#     """
#     H, W, _ = flow1.shape
#     flow1 = np.transpose(flow1, (2, 0, 1))
#     flow2 = np.transpose(flow2, (2, 0, 1)) 
#     assert(flow1.shape == flow2.shape)
#     indices = np.zeros_like(flow1)
#     indices[0] = np.arange(flow1.shape[1])[:, None] + flow1[0]
#     indices[1] = np.arange(flow2.shape[2])[None, :] + flow2[1]
#     #indices: (2, H, W) with absolute flow mapping
    
#     # print(flow1.shape, flow2.shape, indices.shape)
#     indices = indices.reshape(2, -1).astype(np.int64)
#     output_flow = flow1 + flow2[:, indices[0], indices[1]].reshape(2, H, W)
#     # output_flow = flow1 + flow2[indices]
#     return np.transpose(output_flow, (1, 2, 0))

def mergeFlow(flow1, flow2):
    H, W, _ = flow1.shape
    flow1 = np.transpose(flow1, (2, 0, 1))
    flow2 = np.transpose(flow2, (2, 0, 1)) 
    assert(flow1.shape == flow2.shape)
    indices = np.zeros_like(flow1)
    # print(flow1[1])
    indices[0] = np.clip(
        np.arange(flow1.shape[1])[:, None] + flow1[1], 0, H-1
    )
    indices[1] = np.clip(
        np.arange(flow1.shape[2])[None, :] + flow1[0], 0, W-1
    )
    
    
    # print(indices[0])
    # print(indices[1])
    # print("flow1: ", flow1)
    # print("flow2: ", flow2)
    # print("indices: ", indices)
    indices = indices.reshape(2, -1).astype(np.int64)
    correspondingFlows = flow2[:, indices[0], indices[1]].reshape(2, H, W)
    output_flow = flow1 + correspondingFlows
    # print("mask: ", np.all(flow1==0, axis=0))
    output_flow[:, np.all(flow1==0, axis=0)] = 0
    output_flow[:, np.all(correspondingFlows==0, axis=0)] = 0
    # print(output_flow)
    # print("indices: ", indices)
    # print("output flow: ", output_flow)
    return np.transpose(output_flow, (1, 2, 0))

def backpropFlow(flow, im):
    """
    returns im backpropped as if it was im1
    flow: H, W, 2
    im: H, W, 3
    """
    assert(flow.shape[:2] == im.shape[:2])
    H, W, _ = flow.shape
    flow = np.transpose(flow, (2, 0, 1))
    im = np.transpose(im, (2, 0, 1)) 

    indices = np.zeros_like(flow)
    # out_im = np.zeros_like(im)
    
    # print(flow1[1])
    indices[0] = np.clip(
        np.arange(flow.shape[1])[:, None] + flow[1], 0, H-1
    )
    indices[1] = np.clip(
        np.arange(flow.shape[2])[None, :] + flow[0], 0, W-1
    )
    
    
    # print(indices[0])
    # print(indices[1])
    # print("flow1: ", flow1)
    # print("flow2: ", flow2)
    # print("indices: ", indices)
    indices = indices.reshape(2, -1).astype(np.int64)
    output_im = im[:, indices[0], indices[1]].reshape(-1, H, W)

    # print("mask: ", np.all(flow1==0, axis=0))
    output_im[:, np.all(flow==0, axis=0)] = 0
    # print(output_flow)
    # print("indices: ", indices)
    # print("output flow: ", output_flow)
    return np.transpose(output_im, (1, 2, 0))

def imageMap(label, label_map):
    """
    label: (H, W, C)
    label_map: [[r, g, b], index] or [oldIndex, index]
    """
    label = label.transpose((2, 0, 1))
    C, H, W = label.shape
    assert(C<=3)
    if label_map is not None:
        label_copy = label.copy()
        label = np.zeros_like(label)
        for old_id, new_id in label_map:
            # label[label_copy == old_id] = new_id
            mask = np.all(label_copy == np.array(old_id)[:, None, None], axis=0)
            label[:, mask] = new_id
    
    return label.transpose((1, 2, 0))#[:, :, [0]]