import json
import time
import requests
import numpy as np
import torch
from PIL import Image
from io import BytesIO

def get_image_url(image_bytes, upload_url="https://image-upload.catalogix.ai"):
    try:
        files = {'image': ("upload.png", image_bytes, "image/png")}
        response = requests.post(upload_url, files=files, timeout=60)
        response.raise_for_status()
        return response.json()['data']['cloudfront_url']
    except Exception as e:
        raise RuntimeError(f"Image upload failed: {str(e)}")


class NanoBananaBasicNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "replicate_key": ("STRING", {"multiline": False}),
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
    CATEGORY = "Replicate"

    def __init__(self):
        self.api_url = "https://api.replicate.com/v1/models/google/nano-banana-pro/predictions"
        self.poll_base = "https://api.replicate.com/v1/predictions"
        self.poll_interval = 2
        self.max_wait = 300

    def tensor_to_png_bytes(self, tensor_img: torch.Tensor) -> bytes:
        arr = (tensor_img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        arr = arr[:, :, :3]
        img = Image.fromarray(arr)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def poll_prediction(self, replicate_key: str, prediction_id: str):
        headers = {"Authorization": f"Bearer {replicate_key}"}
        start = time.time()
        while True:
            resp = requests.get(f"{self.poll_base}/{prediction_id}", headers=headers, timeout=60)
            data = resp.json()
            status = data.get("status")

            if status in ("succeeded", "failed", "canceled"):
                return data
            if time.time() - start > self.max_wait:
                raise RuntimeError("Prediction timed out")

            time.sleep(self.poll_interval)

    def run(self, replicate_key, prompt, image_input,
            aspect_ratio, resolution, output_format, safety_filter_level):

        # Convert batch tensor images to URLs
        batch, _, _, _ = image_input.shape
        image_urls = []
        for i in range(batch):
            bytes_png = self.tensor_to_png_bytes(image_input[i])
            url = get_image_url(bytes_png)
            image_urls.append(url)

        payload = {
            "input": {
                "prompt": prompt,
                "resolution": resolution,
                "image_input": image_urls,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "safety_filter_level": safety_filter_level,
            }
        }

        headers = {
            "Authorization": f"Bearer {replicate_key}",
            "Content-Type": "application/json"
        }

        # Fire prediction request with retries
        max_attempts = 5
        res = None
        status = None
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                # 1) Create prediction
                r = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                print(f"Replicate request attempt: {attempt}")
                r.raise_for_status()
                res = r.json()

                prediction_id = res.get("id")
                status = res.get("status")

                if not prediction_id:
                    raise RuntimeError("Replicate did not return a prediction id")

                # 2) Poll prediction if not finished yet
                if status not in ("succeeded", "failed", "canceled"):
                    res = self.poll_prediction(replicate_key, prediction_id)
                    status = res.get("status")

                # 3) If succeeded, we're done
                if status in ["succeeded","canceled"]:
                    break

                # If we reach here: status is failed/canceled or something unexpected
                last_error = RuntimeError(f"Replicate status: {status}")

            except Exception as e:
                last_error = e

            # If not last attempt, sleep a bit and retry from scratch
            if attempt < max_attempts:
                time.sleep(2)

        # After all attempts, still not succeeded
        if status != "succeeded":
            raise RuntimeError(
                f"Replicate failed after {max_attempts} attempts. "
                f"Last status: {status}, last_error: {last_error}, last_response: {res}"
            )

        # Get first output URL
        print("RESULT:",res)
        first_url = res["output"]
        out_img_data = requests.get(first_url, timeout=120).content

        # Convert to Tensor
        img = Image.open(BytesIO(out_img_data)).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = img[None, ...]  # Add batch dim

        return (torch.from_numpy(img),)


NODE_CLASS_MAPPINGS = {
    "NanoBananaBasicNode": NanoBananaBasicNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaBasicNode": "Nano Banana Basic"
}
