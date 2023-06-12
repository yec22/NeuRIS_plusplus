from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
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

def show_mask(mask, img, score):
    color = np.clip((np.random.random(3) * 255.), 0, 255).astype(np.uint8)
    img[mask] = color
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"dataset/indoor/scene0050_00/mask_object1/mask_{score}.png", img)

    mask_img = np.zeros_like(img)
    mask_img[mask] = 1.
    mask_img = np.clip((mask_img * 255.), 0, 255).astype(np.uint8)
    cv2.imwrite(f"dataset/indoor/scene0050_00/mask_object1/mask.png", mask_img)



if __name__ == "__main__":
    image = cv2.imread('dataset/indoor/scene0050_00/image_denoised_cv07211010/0800.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = "segment-anything/ckpt/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda:1"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    input_point = np.array([[275, 239], [339, 183], [286, 406], [364, 391]])
    input_label = np.array([1, 1, 1, 1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    show_mask(masks[2], image, scores[2])