from tkinter import W
# import png
import array
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from benchmark_viper import VIPER
from tools.aggregate_flows.flow.util_flow import ReadKittiPngFile
import torch
from torchvision import transforms

def flow_to_grayscale(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = torch.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = torch.sqrt(torch.square(u) + torch.square(v))
    rad_max = torch.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    # print(u.shape, v.shape)
    return torch.stack([u, v])
    # print('hi')

def multiBarChart(data, labels, title="title", xlabel="xlabel", ylabel="ylabel", ax=None, colors=None, figsize=(10, 5), save_path=None):
    """
    Args:
        data: dict of lists of data to plot
        labels: list of labels for each data list
        title: title of plot
        xlabel: x axis label
        ylabel: y axis label
        colors: list of colors
        figsize: size of figure
        save_path: path to save figure
    """
    plotter = ax if ax is not None else plt
    legend = list(data.keys())
    data = list(data.values())

    if colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # assert len(data) == len(labels)
    # assert len(data) == len(legend)
    # assert len(data) <= len(colors)
    
    # Create plot
    # fig, ax = plt.subplots(figsize=figsize)
    index = np.arange(len(data[0]))
    bar_width = 0.15
    opacity = 0.8

    
    for i in range(len(data)):
        plotter.bar(index + bar_width * i, data[i], bar_width,
                 alpha=opacity,
                 color=colors[i],
                 label=legend[i])

    if ax is None:
        plotter.xlabel(xlabel)
        plotter.ylabel(ylabel)
        plotter.title(title)
        plotter.xticks(index + bar_width, labels, rotation=45)
        plotter.legend()
    else:
        plotter.set_xlabel(xlabel)
        plotter.set_ylabel(ylabel)
        plotter.set_title(title)
        plotter.set_xticks(index + bar_width)
        plotter.set_xticklabels(labels, rotation=45)
        plotter.legend()
    
    
    # if save_path is not None:
    #     plotter.savefig(save_path)
    # plotter.show()

def visFlow(flow, image=None, threshold=2.0, skip_amount=30):
    """
    args:
        flow: H, W, 2
        image: H, W, 3
    returns:
        image
    """
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
    img = cv2.resize(img, tuple(out_dim[[1, 0]]))
    # img = img[::2, ::2]
    # img = np.resize(img, out_dim)
    
    _,ret = cv2.imencode('.jpg', img) 
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)

def loadFlow(im_path):
    """
    Args
        im_path: path of flow to load.
    Returns
        flow with shape H, W, 2 (I think)
    """
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

def errorVizClasses(prediction, gt):
    """
    prediction: H, W
    gt: H, W
    Returns H, W displaying the ground truth image whereever the prediction is wrong
    """
    assert(prediction.shape == gt.shape)
    H, W = prediction.shape
    out = np.ones((H, W))*255
    out[gt != prediction] = gt[gt != prediction]
    # out[gt == prediction] = prediction[gt == prediction]
    return out

def backpropFlow(flow_orig, im_orig, return_mask_count=False, return_mask=False):
    """
    returns im backpropped as if it was im1
    flow: torch.Tensor H, W, 2 or B, H, W, 2
    im: torch.Tensor H, W, 3 or B, H, W, 3
    """
    if len(flow_orig.shape) == 3:
        return backpropFlowHelper(flow_orig, im_orig, return_mask_count=return_mask_count, return_mask=return_mask)
    elif len(flow_orig.shape) == 4:
        # shape 0 is batch size
        output = []
        for i in range(flow_orig.shape[0]):
            output.append(backpropFlowHelper(flow_orig[i], im_orig[i], return_mask_count=False, return_mask=False))
        
        return torch.stack(output)
        



def backpropFlowHelper(flow_orig, im_orig, return_mask_count=False, return_mask=False):
    """
    returns im backpropped as if it was im1
    flow: torch.Tensor H, W, 2
    im: torch.Tensor H, W, 3
    """
    flow = flow_orig.clone()
    im = im_orig.clone()

    assert flow.device == im.device, "flow and im must be on the same device"
    dev = flow.device
    assert(flow.shape[:2] == im.shape[:2]) #2048 x 1024
    H, W, _ = flow.shape
    flow = flow.permute(2, 0, 1)
    im = im.permute(2, 0, 1)


    indices = torch.zeros_like(flow).to(dev)
    indices[0] = (torch.arange(flow.shape[1])[:, None]).to(dev) + flow[1]
    indices[1] = (torch.arange(flow.shape[2])[None, :]).to(dev) + flow[0]

    flow[:, indices[0] >= 840] = 0
    flow[:, indices[0] < 0] = 0
    flow[:, indices[1] >= W] = 0
    flow[:, indices[1] < 0] = 0

    indices[0] = (torch.arange(flow.shape[1])[:, None]).to(dev) + flow[1]
    indices[1] = (torch.arange(flow.shape[2])[None, :]).to(dev) + flow[0]
    
    indices = indices.reshape(2, -1).long()
    indices_t = indices.permute((1, 0))
    unique, counts = torch.unique(indices_t, dim=0, return_counts=True)
    mask_indices = unique[counts > 1].permute((1, 0))

    output_im = im[:, indices[0], indices[1]].reshape(-1, H, W)

    output_im[:, mask_indices[0], mask_indices[1]] = 255
    output_im[:, torch.all(flow==0, dim=0)] = 255

    to_return = [output_im.permute((1, 2, 0))]
    
    mask = (output_im!=255).all(dim=0)
    if return_mask:
        to_return.append(mask)
    
    if return_mask_count:
        unique, counts = torch.unique(im[:, ~mask], return_counts=True)
        total_unique, total_counts = torch.unique(im, return_counts=True)
        mask_count = [unique.cpu(), counts.cpu(), total_unique.cpu(), total_counts.cpu()]
        to_return.append(mask_count)
    
    if len(to_return) == 1:
        return to_return[0]
    else:
        return to_return

def backpropFlowNoDup(flow, im_orig, return_mask_count=False, return_mask=False):
    """
    returns im t+k backpropped as if it was im t
    flow: H, W, 2
    im: H, W, 3
    """
    im = im_orig.copy()

    assert(flow.shape[:2] == im.shape[:2])
    H, W, _ = flow.shape
    mask = np.ones(flow.shape[:2])
    # TODO: this should dynamically crop off the bottom % of the image, not just >= 920
    flow[920:, :, :] = 0
    flow = np.transpose(flow, (2, 0, 1))
    im = np.transpose(im, (2, 0, 1))

    indices = np.zeros_like(flow)
    indices[0] = np.arange(flow.shape[1])[:, None] + flow[1]
    indices[1] = np.arange(flow.shape[2])[None, :] + flow[0]

    flow[:, indices[0] > 920] = 0
    flow[:, indices[0] < 0] = 0
    flow[:, indices[1] >= H] = 0
    flow[:, indices[1] < 0] = 0

    indices[0] = np.arange(flow.shape[1])[:, None] + flow[1]
    indices[1] = np.arange(flow.shape[2])[None, :] + flow[0]
    
    indices = indices.reshape(2, -1).astype(np.int64)

    indices_t = indices.transpose((1, 0))
    unique, counts = np.unique(indices_t, axis=0, return_counts=True)
    mask_indices = unique[counts > 1].transpose((1, 0))
    # print(np.sum(unique[counts > 1]))
    # print(mask_indices)
    if return_mask_count:
        unique, counts = np.unique(im[:, mask_indices[0], mask_indices[1]], return_counts=True)
        total_unique, total_counts = np.unique(im, return_counts=True)
        mask_count = [unique, counts, total_unique, total_counts]
    
    im[:, mask_indices[0], mask_indices[1]] = 255 #np.array([255, 192, 203]).reshape(3, 1)

    output_im = im[:, indices[0], indices[1]].reshape(-1, H, W)
    output_im[:, np.all(flow==0, axis=0)] = 255

    # mask[mask_indices[0], mask_indices[1]] = 0
    # mask[np.all(flow==0, axis=0)] = 0


    to_return = [np.transpose(output_im, (1, 2, 0))]


    if return_mask_count:
        to_return.append(mask_count)
    
    if return_mask:
        # if (output_im == 255).shape[0] != 1:
            # assert False, "return mask not implemented for images with multiple channels"
        # mask = (output_im != 255).squeeze(0)
        mask = (output_im!=255).all(axis=0)
        to_return.append(mask)
    
    return to_return[0] if len(to_return) == 1 else tuple(to_return)

def backpropFlowFilter2(flow, im2, thresh):
    """
    returns im backpropped as if it was im1
    flow: H, W, 2
    im: H, W, 3
    """
    norm = np.linalg.norm(flow, axis=2)
    new_flow = flow.copy()
    new_flow[norm < thresh] = 0
    im2_1 = backpropFlow(new_flow, im2)

    return im2_1

def backpropFlowFilter(flow, im2, im1, thresh=300):
    """
    returns im backpropped as if it was im1
    flow: H, W, 2
    im2: H, W, 3
    im1: H, W, 3
    thresh: max norm diff b/w original and back propped point before filtering out
    """
    im2_1 = backpropFlow(flow, im2)
    print("filter shapes: ", im1.shape, im2.shape)

    diff = np.linalg.norm(im2_1 - im1, axis=2)
    print(diff.mean(), diff.std())
    mask = diff > thresh
    print("mask shape: ", mask.shape, im2_1.shape)
    im2_1[mask] = 0

    return im2_1

def tensor_map(tensor, label_map):
    """
    tensor: any shape tensor
    label_map: {old: new}
    """
    tensor_copy = tensor.clone()
    for old_id, new_id in label_map.items():
        tensor_copy[tensor == old_id] = new_id
    
    return tensor_copy

def imageMap2(label, label_map):
    """
    label: (H, W, C) numpy array
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
            label[:, mask] = np.array(new_id)[:, None]
    
    return label.transpose((1, 2, 0))#[:, :, [0]]


def imageMap(label, label_map):
    """
    label: (H, W, C) numpy array
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

def labelMapToIm(label, label_map):
    """
    label: H, W, 1 tensor
    label_map: [[r, g, b], index]]
    """
    output = label.repeat(1, 1, 3)
    for color, id in label_map:
        output[label.squeeze(2)==id] = torch.tensor(color)
    return output

palette_to_id = [   
    ([0,0,0], 0),
    ([111,74,0], 1),
    ([70,130,180], 2),
    ([128,64,128], 3),
    ([244,35,232], 4),
    ([230,150,140], 5),
    ([152,251,152], 6),
    ([87,182,35], 7),
    ([35,142,35], 8),
    ([70,70,70], 9),
    ([153,153,153], 10),
    ([190,153,153], 11),
    ([150,20,20], 12),
    ([250,170,30], 13),
    ([220,220,0], 14),
    ([180,180,100], 15),
    ([173,153,153], 16),
    ([168,153,153], 17),
    ([81,0,21], 18),
    ([81,0,81], 19),
    ([220,20,60], 20),
    ([255,0,0], 21),
    ([119,11,32], 22),
    ([0,0,230], 23),
    ([0,0,142], 24),
    ([0,80,100], 25),
    ([0,60,100], 26),
    ([0,0,70], 27),
    ([0,0,90], 28),
    ([0,80,100], 29),
    ([0,100,100], 30),
    ([50,0,90], 31)
]

def rare_class_or_filter(pl1, pl2, rare_common_compare=False):
        """
        pl1: (B, H, W)
        pl2: (B, H, W)
        rare_common_compare: boolean that determines whether to do priority of rarity between classes, or just abs if it belongs to "rare" or "common" group
        returns a pseudolabel which keeps consistent pixels, and masks out inconsistent pixels except when the pixel is rare.  In the case that both pl1 and pl2 are rare take the more rare pixel
        """
        # most to least rare
        # pl1[pl1 == 255]
        #                                                     | 
        rarity_order = [3, 5, 12, 16, 18, 17, 7, 6, 15, 4, 11, 14, 9, 8, 1, 2, 10, 13, 0]
        # rarity_index = torch.tensor([rarity_order.index(i) for i in range(19)])
        rarity_thresh = 11 #rarity_order[:11] is rare, past that not rare

        inconsis_pixels = pl1 != pl2
        consistent_pixels = pl1 == pl2
        # pl1_rarity = rarity_index[]
        pl1_rarity = tensor_map(pl1, {i: rarity_order.index(i) for i in range(19)}) # {0: 19, 1:15} ie 0 is the 19th rarest class, 1 is the 15th rarest class
        pl2_rarity = tensor_map(pl2, {i: rarity_order.index(i) for i in range(19)})
        output = torch.ones_like(pl1)*255
        # print("pl1 rarity", pl1_rarity)
        # print("pl2 rarity", pl2_rarity)

        # print("output1", output)
        output[consistent_pixels] = pl1[consistent_pixels]
        # print("output2", output)
        
        if rare_common_compare:
            #only take prediction if one is rare and other is common
            pl1_rarer_and_inconsistent = inconsis_pixels & (pl1_rarity < rarity_thresh) & (pl2_rarity >= rarity_thresh)
            output[pl1_rarer_and_inconsistent] = pl1[pl1_rarer_and_inconsistent]

            pl2_rarer_and_inconsistent = inconsis_pixels & (pl2_rarity < rarity_thresh) & (pl1_rarity >= rarity_thresh)
            output[pl2_rarer_and_inconsistent] = pl2[pl2_rarer_and_inconsistent]

        else:
            pl1_rarer_and_inconsistent = inconsis_pixels & (pl1_rarity < rarity_thresh) & (pl1_rarity < pl2_rarity)
            output[pl1_rarer_and_inconsistent] = pl1[pl1_rarer_and_inconsistent]

            pl2_rarer_and_inconsistent = inconsis_pixels & (pl2_rarity < rarity_thresh) & (pl2_rarity < pl1_rarity)
            output[pl2_rarer_and_inconsistent] = pl2[pl2_rarer_and_inconsistent]
        # breakpoint()

        return output

invNorm = transforms.Compose([
    transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]), #Using some other dataset mean and std
    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])
])

class CircularTensor:
    def __init__(self, max_length):
        self.max_length = max_length
        self.buffer = torch.zeros(max_length, dtype=torch.double)
        self.idx = 0
        self.full = False
    def append(self, data):
        if data.ndim > 1:
            raise Exception("Passed in data does not have ndim = 1")
        
        if data.size()[0] > self.max_length:
            raise Exception("Data length is greater than max_length")
        
        # now dealign with adding to list
        
        idx_start = self.idx % self.max_length
        idx_end = (self.idx + data.size()[0]) % self.max_length

        if idx_start >= idx_end:
            if not self.full:
                self.full = True

            len1 = self.max_length - idx_start
            self.buffer[idx_start: self.max_length] = data[:len1]
            self.buffer[:idx_end] = data[len1:]
        else:
            self.buffer[idx_start: idx_end] = data
        
        self.idx = idx_end
    
    def get_mean(self):
        
        if self.full:
            return torch.mean(self.buffer)
        else:
            if self.idx == 0:
                return 0
            return torch.mean(self.buffer[:self.idx])
    
    def get_buffer(self):
        return self.buffer
    

    

