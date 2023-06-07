from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import cv2


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random(3)
        img[m] = color_mask
    img = np.clip((img * 255.), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("mask.png", img)


if __name__ == "__main__":
    image = cv2.imread('dataset/indoor/scene0616_00/image/0000.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = "segment-anything/ckpt/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda:1"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)

    show_anns(masks)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("image.png", image)