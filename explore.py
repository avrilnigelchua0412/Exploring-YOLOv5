from utils.dataloaders import InfiniteDataLoader, LoadImagesAndLabels
import os
import random
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

def seed_worker(worker_id):
    """
    Sets the seed for a dataloader worker to ensure reproducibility, based on PyTorch's randomness notes.

    See https://pytorch.org/docs/stable/notes/randomness.html#dataloader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_unnormalized_bounding_box(x_center, y_center, w_norm, h_norm, img_width, img_height):
    # Step 1: Convert normalized width/height back to pixels
    bbox_width = w_norm * img_width
    bbox_height = h_norm * img_height
    
    # Step 2: Convert center coords to top-left corner
    # Multiply normalized center by image size to get absolute pixel center
    abs_x_center = x_center * img_width
    abs_y_center = y_center * img_height
    
    # Subtract half the width/height to shift from center → top-left
    x_min = abs_x_center - (bbox_width / 2)
    y_min = abs_y_center - (bbox_height / 2)
    
    return x_min, y_min, bbox_width, bbox_height

def visualize_bboxes(bboxes, labels, ax):
    """Draw bounding boxes with color-coded labels on a Matplotlib axis."""
    for box, label in zip(bboxes, labels):
        x_min, y_min, box_width, box_height = box
        # choose color based on label
        color = "red" if label == 'Cluster' else "blue"
        # choose linewidth
        linewidth = 3 if label == 'Cluster' else 2
        # draw bounding box
        ax.add_patch(plt.Rectangle(
            (x_min, y_min),
            box_width, box_height,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none"
        ))
        # draw label text
        ax.text(
            x_min, y_min - 5,
            label,
            color=color,
            fontsize=10,
            weight="bold"
        )
        
if __name__ == "__main__":
    map = {0: 'Cluster', 1 : 'Thyrocyte'}
    path = '/workspace/Special_Problem/yolo_dataset_version_2/images/train'
    dataset = LoadImagesAndLabels(
        path=path,
        img_size=512,
        batch_size=1,
        augment=False,
        rect=False,
        cache_images=False,
        single_cls=False
    )
    num_classes = int(max([max(l[:, 0]) if len(l) > 0 else 0 for l in dataset.labels]) + 1)
    print("Number of data samples:", len(dataset))
    N = len(dataset)
    class_counts = torch.zeros(num_classes)
    for labels in dataset.labels:
        for label in labels:
            class_id = int(label[0])
            class_counts[class_id] += 1

    print("Class counts:", class_counts)
    
    # Inverse frequency weights per class
    class_weights = 1.0 / torch.clamp(class_counts, min=1)
    print("Class weights:", class_weights)
    
    # Normalized Inverse frequency weights per class
    norm_class_weights = N / torch.clamp((num_classes * class_counts), min=1)
    print("Normalized Inverse Frequency:", norm_class_weights)

    # Assign each image a weight based on its labels
    sample_weights = []
    norm_sample_weights = []
    for labels in dataset.labels:
        if len(labels) == 0:
            sample_weights.append(0)  # no labels (background)
        else:
            
            # average of label weights for that image
            img_weight = []
            norm_img_weight = []
            for l in labels:
                # print(l)
                img_weight.append(class_weights[int(l[0])])
                norm_img_weight.append(norm_class_weights[int(l[0])])
                
            sample_weights.append(np.mean(img_weight))
            norm_sample_weights.append(np.mean(norm_img_weight))

    
    # Compute sampling probabilities
    sample_weights_np = np.array(sample_weights)
    sampling_probs = sample_weights_np / sample_weights_np.sum()
    norm_sample_weights_np = np.array(norm_sample_weights)
    norm_sampling_probs = norm_sample_weights_np / norm_sample_weights_np.sum()

    plt.figure(figsize=(8, 3))
    plt.bar(range(len(sampling_probs)), sampling_probs)
    plt.title("Sampling Probability per Image")
    plt.xlabel("Image Index")
    plt.ylabel("Probability")
    plt.show()
    
    plt.figure(figsize=(8, 3))
    plt.bar(range(len(norm_sampling_probs)), norm_sampling_probs)
    plt.title("Normalized Sampling Probability per Image")
    plt.xlabel("Image Index")
    plt.ylabel("Probability")
    plt.show()
    # # sample_weights = torch.DoubleTensor(sample_weights)
    
    # sampler = WeightedRandomSampler(
    #     weights=sample_weights,
    #     num_samples=len(sample_weights),  # same length as dataset
    #     replacement=True                  # allow resampling same image
    # )
    
    # # for index in sampler:
    # #     image = dataset[index][0].permute(1, 2, 0).cpu().numpy()
    # #     image_shape, (scaling_factors, padding_offsets) = dataset[index][3]
    # #     (img_height, img_width) = image_shape
    # #     fig, ax = plt.subplots(1, figsize=(5, 5))
    # #     plt.imshow(image)
    # #     bboxes = []
    # #     labels = []
    # #     for bbox in dataset[index][1].cpu().tolist():
    # #         cls, x_center, y_center, w_norm, h_norm = bbox[1:] # after collate_fn4 hence index 0 should be exlucded
    # #         bboxes.append(get_unnormalized_bounding_box(x_center, y_center, w_norm, h_norm, img_width, img_height))
    # #         labels.append(map[cls])
    # #     visualize_bboxes(bboxes, labels, ax)
    # #     plt.show()
    
    # loader = InfiniteDataLoader(
    #     dataset,
    #     batch_size=16,
    #     shuffle=False,
    #     num_workers=8,
    #     sampler=sampler,
    #     drop_last=False,
    #     pin_memory=str(os.getenv("PIN_MEMORY", True)).lower() == "true",
    #     collate_fn=LoadImagesAndLabels.collate_fn,
    #     worker_init_fn=seed_worker,
    #     generator=torch.Generator(),
    # )
    
    # for i, (imgs, targets, paths, shapes) in enumerate(loader):
    #     print(f"Batch {i}:")
    #     print(f"  Image tensor shape: {imgs.shape}")
    #     print(f"  Targets shape: {targets.shape}")
    #     print(f"  Example paths: {[os.path.basename(p) for p in paths]}")
    #     print(f"  Example label rows: {targets[:5]}")
    #     break