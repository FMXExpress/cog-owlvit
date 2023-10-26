import os
from typing import Dict, Optional

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from cog import BasePredictor, BaseModel, Input, Path
from file_utils import download_weights


WEIGHTS_CACHE_DIR = "/src/owl-vit-cache/"
WEIGHTS_URL = "https://weights.replicate.delivery/default/owlvit/owl-vit-patch-16.tar"

if not os.path.exists(WEIGHTS_CACHE_DIR):
    download_weights(url=WEIGHTS_URL,dest=WEIGHTS_CACHE_DIR)

os.environ["HF_HOME"] = os.environ["HUGGINGFACE_HUB_CACHE"] = WEIGHTS_CACHE_DIR

class ModelOutput(BaseModel):
    json_data: Dict
    result_image: Optional[Path]


class Predictor(BasePredictor):
    def setup(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.processor = OwlViTProcessor.from_pretrained(WEIGHTS_CACHE_DIR)
        self.model = OwlViTForObjectDetection.from_pretrained(WEIGHTS_CACHE_DIR).to(self.device)
        self.model.eval()
    
    def query_image(
        self, 
        image_path, 
        text_queries, 
        score_threshold, 
        show_visualisation
    ): 
        img = Image.open(str(image_path)).convert("RGB")
        target_sizes = torch.Tensor([img.size[::-1]])

        text_queries = text_queries
        text_queries = text_queries.split(",")
        inputs = self.processor(text=text_queries, images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        outputs.logits = outputs.logits.cpu()
        outputs.pred_boxes = outputs.pred_boxes.cpu() 
        results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        slice = scores>=score_threshold
        boxes, scores, labels = boxes[slice,...], scores[slice,...], labels[slice,...]

        #font = cv2.FONT_HERSHEY_SIMPLEX

        json_data = {"objects": [],}  # label, confidence, bbox
        
        #res_img = np.array(img)[:,:,::-1] # numpy image to draw on via cv2. reorder channels!
        res_img = np.zeros_like(img)
        result_img_path = "/tmp/result.png" if show_visualisation else None

        res_img[:] = (0, 0, 0)
        for box, score, label in zip(boxes, scores, labels):
            box = [int(i) for i in box.tolist()]

            data = {
                "label": text_queries[label].strip(),
                "confidence": score.item(), #torch tensor to float
                "bbox": box,
            }
            json_data["objects"].append(data)
        
            if show_visualisation:
                res_img = res_img.copy()
                # Draw white filled rectangles
                res_img = cv2.rectangle(res_img, box[:2], box[2:], (255, 255, 255), -1)
                
                #res_img = res_img.copy()
                #res_img = cv2.rectangle(res_img, box[:2], box[2:], (255,0,0), 2)
                #if box[3] + 25 > 768:
                #    y = box[3] - 10
                #else:
                #    y = box[3] + 25

                #res_img = cv2.putText(
                #    res_img, text_queries[label], (box[0], y), font, 1, (255,0,0), 2, cv2.LINE_AA
                #)

        if show_visualisation:
            cv2.imwrite(result_img_path, res_img)

        return result_img_path, json_data

    def predict(
        self,
        image: Path = Input(
            description="Input image to query", default=None
        ),
        query: str = Input(
            description="Comma seperated names of the objects to be detected in the image", default=None
        ),
        threshold: float = Input(
            description="Confidence level for object detection", ge=0, le=1, default=0.1
        ),
        show_visualisation: bool = Input(
            description="Draw and visualize bounding boxes on the image", default=True
        )
    ) -> ModelOutput:

        result_img_path, json_data = self.query_image(image, query, threshold, show_visualisation)
        
        return ModelOutput(
            json_data=json_data, 
            result_image=Path(result_img_path) if show_visualisation else None
        )
