from my_utils import loadFlow, visFlow, backpropFlowNoDup
import cv2


img1_pth = "/srv/share4/datasets/cityscapes-seq/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_000275_leftImg8bit.png"
img2_pth = "/srv/share4/datasets/cityscapes-seq/leftImg8bit_sequence/val/frankfurt/frankfurt_000000_000276_leftImg8bit.png"
flow_pth = "/srv/share4/datasets/cityscapes-seq_Flow/flow/forward/val/frankfurt/frankfurt_000000_000275_leftImg8bit.png"

flow = loadFlow(flow_pth)
img1 = cv2.imread(img1_pth)
img2= cv2.imread(img2_pth)



img1_name = img1_pth.split('/')[-1]
img2_name = img2_pth.split('/')[-1]
cv2.imwrite(img1_name, img1)
# cv2.imwrite(img2_name, img2)

flow_name = "flow_arrow_" +flow_pth.split('/')[-1]
img_flow_arrow = visFlow(flow, img1, skip_amount=350)
cv2.imwrite(flow_name, img_flow_arrow)


backprop_img = backpropFlowNoDup(flow, img2)

backprop_flow_name = "backprop_flow_" +flow_pth.split('/')[-1]
cv2.imwrite(backprop_flow_name, backprop_img)