import os

import cv2
import numpy as np
import torch
from PIL import Image

# Make Numba shut up.
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"
os.environ["NUMBA_DISABLE_TBB"] = "1"
os.environ["NUMBA_THREADING_LAYER"] = "omp"

import rembg

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']


class BLIP2():
    def __init__(self, device='cuda'):
        self.device = device
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",
                                                                   torch_dtype=torch.float16).to(device)

    @torch.no_grad()
    def __call__(self, image):
        image = Image.fromarray(image)
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return generated_text


def process(image: Image, model: str = 'u2net', size: int = 512, border_ratio: float = 0.2,
            recenter: bool = True):
    session = rembg.new_session(model_name=model, providers=providers)

    image = np.array(image)

    # carve background
    print(f'[INFO] background removal...')
    carved_image = rembg.remove(image, session=session)  # [H, W, 4]
    mask = carved_image[..., -1] > 0

    # recenter
    if recenter:
        print(f'[INFO] recenter...')
        final_rgba = np.zeros((size, size, 4), dtype=np.uint8)

        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        h = x_max - x_min
        w = y_max - y_min
        desired_size = int(size * (1 - border_ratio))
        scale = desired_size / max(h, w)
        h2 = int(h * scale)
        w2 = int(w * scale)
        x2_min = (size - h2) // 2
        x2_max = x2_min + h2
        y2_min = (size - w2) // 2
        y2_max = y2_min + w2
        final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(carved_image[x_min:x_max, y_min:y_max], (w2, h2),
                                                              interpolation=cv2.INTER_AREA)
    else:
        final_rgba = carved_image

    return Image.fromarray(final_rgba)
