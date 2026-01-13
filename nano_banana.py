import json
import time
import requests
import numpy as np
import torch
from PIL import Image
from io import BytesIO


class NanoBananaBasicNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gemini_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True}),
                "image_input": ("IMAGE",),
                "aspect_ratio": ([
                    "match_input_image",
                    "1:1", "2:3", "3:2", "3:4",
                    "4:3", "4:5", "5:4",
                    "9:16", "16:9", "21:9"
                ],),
                "resolution": (["1K", "2K", "4K"],),
                "output_format": (["png", "jpg"],),
                "safety_filter_level": ([
                    "block_only_high",
                    "block_medium_and_above",
                    "block_low_and_above"
                ],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Gemini"

    def __init__(self):
        self.api_url = (
            "https://generative-api-132358776415.asia-south1.run.app/"
            "generative-service/gemini/nanobananapro/generate"
        )

    def tensor_to_png_bytes(self, tensor_img: torch.Tensor) -> bytes:
        arr = (tensor_img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        arr = arr[:, :, :3]
        img = Image.fromarray(arr)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def run(
        self,
        gemini_key,
        prompt,
        image_input,
        aspect_ratio,
        resolution,
        output_format,
        safety_filter_level
    ):
        # Convert batch tensor images to PNG bytes
        batch, _, _, _ = image_input.shape
        files = []

        for i in range(batch):
            png_bytes = self.tensor_to_png_bytes(image_input[i])
            files.append(
                ("images", (f"image_{i}.png", png_bytes, "image/png"))
            )

        # Form data
        data = {
            "prompt": prompt,
            "aspect_ratio": "custom" if aspect_ratio == "match_input_image" else aspect_ratio,
            "resolution": "match_input",
            "format": output_format.upper(),
        }

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {gemini_key}",
        }

        # Call Gemini Nano Banana Pro API
        response = requests.post(
            self.api_url,
            headers=headers,
            files=files,
            data=data,
            timeout=360,
        )
        response.raise_for_status()

        # API returns raw image bytes
        out_img_data = response.content

        # Convert output image to tensor
        img = Image.open(BytesIO(out_img_data)).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = img[None, ...]  # Add batch dimension

        return (torch.from_numpy(img),)


NODE_CLASS_MAPPINGS = {
    "NanoBananaBasicNode": NanoBananaBasicNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaBasicNode": "Nano Banana Basic"
}
