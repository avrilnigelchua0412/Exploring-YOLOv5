from ultralytics import YOLO
import torch
from models.common import DetectMultiBackend
import cv2
from utils.general import non_max_suppression
from ensemble_boxes import weighted_boxes_fusion
import matplotlib.pyplot as plt

class InferenceModel:
    def __init__(self, model_path: str, img_path: str):
        self.model_path = model_path
        self.img_path = img_path
        
        self.model = None
        self.device = None
        self.img = None
        self.img_tensor = None
        self.pred = None
        self.train_out = None
        self.image_tile_metadata = None
        self.tiles = None
        self.model_device = None
        
        self._setup()

    def _setup(self):
        self.set_device()
        self.load_model()
        self.load_image()
        self.prepare_tensor()
        
    def set_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_device(self):
        return self.device

    def load_model(self):
        self.model = DetectMultiBackend(
            self.model_path, 
            device=self.device
        )
        self.model.eval()

    def get_model(self):
        return self.model
    
    def load_image(self):
        img = cv2.imread(self.img_path)
        if img is None:
            raise ValueError(f"Image not found: {self.img_path}")
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_image(self):
        return self.img

    def prepare_tensor(self):
        img_tensor = torch.from_numpy(self.img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        # self.img_tensor = img_tensor.unsqueeze(0).to(self.get_device()) # add batch dim
        self.model_device = next(self.model.parameters()).device
        self.img_tensor = img_tensor.to(self.model_device)

    def get_image_tensor(self):
        return self.img_tensor

    def predict(self):
        if self.model is None:
            raise ValueError("Model is not loaded.")
        if self.img_tensor is None:
            raise ValueError("Image tensor missing.")

        print("Image tensor device:", self.img_tensor.device)
        print("Model device:", next(self.model.parameters()).device)

        # YOLOv5 forward (returns pred, train_output)
        with torch.no_grad():
            self.pred, self.train_out = self.model(
                # self.img_tensor,
                self.get_tiles(),  # Use tiles for inference
                augment=False,
                visualize=False
            )

        # return self.pred, self.train_out

    def set_non_max_suppression(self, conf_thres=0.25, iou_thres=0.45):
        if self.pred is None:
            raise ValueError("Run predict() before NMS.")
        
        self.pred = non_max_suppression(
            self.pred, 
            conf_thres=conf_thres, 
            iou_thres=iou_thres
        )

    def get_non_max_suppression(self):
        return self.pred, self.train_out
    
    def populate_image_tile_metadata(self):
        self.image_tile_metadata = []
        self.tiles = []
        for tile, x0, y0, tile_id in image_tiling(self.img):
            
            self.image_tile_metadata.append({
                'tile_id': tile_id,
                'x0': x0,
                'y0': y0,
                'shape': tile.shape
            })
            self.tiles.append(torch.from_numpy(tile).permute(2, 0, 1).contiguous().to(dtype=torch.float32) / 255.0)
            
    def get_image_tile_metadata(self):
        return self.image_tile_metadata
    
    def get_tiles(self):
        return torch.stack(self.tiles, dim=0).to(self.model_device)

def image_tiling(original_image, tile_size=512, overlap=.25):
        """
        Generate tiles from the original image with specified overlap.
        Yields tuples of (tile, x0, y0, tile_id) where (x0, y0) is the top-left
        coordinate of the tile in the original image and tile_id is a unique identifier.
        """
        stride = int(tile_size * (1 - overlap))
        img_height, img_width, _ = original_image.shape
        for row_idx, y0 in enumerate(range(0, img_height, stride)):
            for col_idx, x0 in enumerate(range(0, img_width, stride)):
                tile_id = f"{row_idx}_{col_idx}"
                x1 = min(x0 + tile_size, img_width)
                y1 = min(y0 + tile_size, img_height)
                tile = original_image[y0:y1, x0:x1]
                tile = pad_image(tile)
                yield tile, x0, y0, tile_id

def pad_image(image):
    img_height, img_width, _ = image.shape
    desired = 512
    
    pad_bottom = max(desired - img_height, 0)
    pad_right = max(desired - img_width, 0)
    if pad_bottom > 0 or pad_right > 0:
        image = cv2.copyMakeBorder(
            image,
            0, pad_bottom, 0, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
    return image
          
def get_image_data(image_path):
        bgr_image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return rgb_image
    
def visualize_bboxes(bboxes, scores, labels, ax):
    """Draw bounding boxes with color-coded labels on a Matplotlib axis."""
    for box, label, score in zip(bboxes[0], labels[0], scores[0]):
        # print(f"Box: {box}, Label: {label}")
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        x_min, y_min = x1, y1
        box_width = x2 - x1
        box_height = y2 - y1
        color = "black"  if label == 0 else "red"
        linewidth = 2
        fontsize = 8
        ax.add_patch(plt.Rectangle(
            (x_min, y_min), box_width, box_height, linewidth=linewidth,
            edgecolor=color, facecolor="none"
        ))
        ax.text(
            x_min, y_min - 5, f"{label}: with {score:.2f}", color=color,
            fontsize=fontsize, weight="bold"
        )

def normalize_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    return [
            (x1) / img_w,
            (y1) / img_h,
            (x2) / img_w,
            (y2) / img_h
        ]

def denormalize_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    return [
            x1 * img_w,
            y1 * img_h,
            x2 * img_w,
            y2 * img_h
        ]

if __name__ == '__main__':
    
    """
    Inference for YOLOv5 model using DetectMultiBackend.
    
    """
    print("Starting inference...")
    MODEL_PATH = 'Exploring-YOLOv5/runs/train/thyro_finetune_phase2/weights/best.pt'
    IMG_PATHS = [
                'Data/BATCH 1/LS-009.jpeg',
                'Data/BATCH 1/LS-018.jpeg',
                'Data/BATCH 1/LS-025.jpg',
                'Data/BATCH 1/LS-032.jpeg',
                'Data/BATCH 1/LS-049.jpeg'
                ]
    MODEL_NAME = MODEL_PATH.split('/')[3]
    for INDEX, IMG_PATH in enumerate(IMG_PATHS):
        BOXES_LIST = []
        SCORES_LIST = []
        LABELS_LIST = []
        
        inference_model = InferenceModel(MODEL_PATH, IMG_PATH)
        inference_model.populate_image_tile_metadata()
        tiles = inference_model.get_tiles()
        
        tile_metadata = inference_model.get_image_tile_metadata()
        
        inference_model.predict()
        inference_model.set_non_max_suppression(conf_thres=0.2, iou_thres=0.2)
        preds, train_out = inference_model.get_non_max_suppression()
        img = inference_model.get_image()
        
        img_h, img_w = img.shape[:2]
        for i, p in enumerate(preds):
            x0 = tile_metadata[i]['x0']
            y0 = tile_metadata[i]['y0']

            if p is None or len(p) == 0:
                continue

            p = p.cpu().numpy()

            for det in p:
                x1, y1, x2, y2, conf, cls = det
                global_box_normalized = normalize_box(
                    (x1 + x0, y1 + y0, x2 + x0, y2 + y0),
                    img_w,
                    img_h
                )

                BOXES_LIST.append(global_box_normalized)
                SCORES_LIST.append(float(conf))
                LABELS_LIST.append(int(cls))
        
        boxes, scores, labels = weighted_boxes_fusion([BOXES_LIST], [SCORES_LIST], [LABELS_LIST], 
            iou_thr=0.55,
            skip_box_thr=0.6)
        
        denormalized_boxes = []
        for box in boxes:
            fused_box_pixel = denormalize_box(box, img_w, img_h)
            denormalized_boxes.append(fused_box_pixel)
        
        fig, ax = plt.subplots(1, figsize=(15, 15))
        ax.imshow(img) 
        visualize_bboxes([denormalized_boxes], [scores], [labels], ax) 
        plt.savefig(f"{MODEL_NAME}-{INDEX}.png")