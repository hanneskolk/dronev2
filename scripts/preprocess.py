import cv2
import os
from tqdm import tqdm

def enhance_image(img):
    # CLAHE contrast boost
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0)
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(os.listdir(input_dir)):
        if not file.endswith(".jpg"):
            continue

        img = cv2.imread(os.path.join(input_dir, file))
        img = enhance_image(img)

        cv2.imwrite(os.path.join(output_dir, file), img)


# Example usage
# process_folder("dataset/train/images", "dataset/train/images_enhanced")