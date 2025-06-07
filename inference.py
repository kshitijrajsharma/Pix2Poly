import argparse
import json
import os
import time
from functools import partial
from glob import glob

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing
from albumentations.pytorch import ToTensorV2
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex
from torchvision.utils import make_grid
from tqdm import tqdm

from config import CFG
from models.model import Decoder, Encoder, EncoderDecoder
from tokenizer import Tokenizer
from utils import permutations_to_polygons, postprocess, test_generate

torch.multiprocessing.set_sharing_strategy("file_system")


def bounding_box_from_points(points):
    points = np.array(points).flatten()
    even_locations = np.arange(points.shape[0] // 2) * 2
    odd_locations = even_locations + 1
    X = np.take(points, even_locations.tolist())
    Y = np.take(points, odd_locations.tolist())
    bbox = [X.min(), Y.min(), X.max() - X.min(), Y.max() - Y.min()]
    bbox = [int(b) for b in bbox]
    return bbox


def single_annotation(image_id, poly):
    _result = {}
    _result["image_id"] = str(image_id)
    _result["category_id"] = 100
    _result["score"] = 1
    _result["segmentation"] = poly
    _result["bbox"] = bounding_box_from_points(_result["segmentation"])
    return _result


class SimpleImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=image)
            image_tensor = transformed["image"]
        else:
            image_tensor = torch.from_numpy(image)

        return image_tensor, os.path.basename(image_path).split(".")[0]


def collate_fn(batch):
    images, image_ids = [], []
    for image, img_id in batch:
        images.append(image)
        image_ids.append(img_id)

    images = torch.stack(images)
    return images, image_ids


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_dir", required=True, help="Directory with input images"
    )
    parser.add_argument(
        "-c", "--checkpoint_path", required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "-o", "--output_dir", default="output", help="Directory to save predictions"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=8, help="Batch size for inference"
    )
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CFG.DEVICE = device

    # Create output directory structure
    experiment_name = os.path.basename(
        os.path.dirname(os.path.dirname(args.checkpoint_path))
    )
    checkpoint_name = os.path.basename(args.checkpoint_path).split(".")[0]

    output_path = os.path.join(
        "runs", experiment_name, "predictions", args.output_dir, checkpoint_name
    )
    os.makedirs(output_path, exist_ok=True)

    # Setup transforms
    transforms = A.Compose(
        [
            A.Resize(height=CFG.INPUT_HEIGHT, width=CFG.INPUT_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    # Initialize tokenizer
    tokenizer = Tokenizer(
        num_classes=1,
        num_bins=CFG.NUM_BINS,
        width=CFG.INPUT_WIDTH,
        height=CFG.INPUT_HEIGHT,
        max_len=CFG.MAX_LEN,
    )
    CFG.PAD_IDX = tokenizer.PAD_code

    # Get images
    image_paths = []
    for ext in ["*.jpg", "*.png", "*.tif", "*.tiff"]:
        image_paths.extend(glob(os.path.join(args.input_dir, ext)))

    if not image_paths:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Found {len(image_paths)} images")

    # Create dataset and dataloader
    dataset = SimpleImageDataset(image_paths, transforms)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=2
    )

    # Initialize model
    encoder = Encoder(model_name=CFG.MODEL_NAME, pretrained=True, out_dim=256)
    decoder = Decoder(
        cfg=CFG,
        vocab_size=tokenizer.vocab_size,
        encoder_len=CFG.NUM_PATCHES,
        dim=256,
        num_heads=8,
        num_layers=6,
    )
    model = EncoderDecoder(cfg=CFG, encoder=encoder, decoder=decoder)
    model.to(device)

    # Load model checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"Model loaded from epoch: {checkpoint['epochs_run']}")

    # Initialize metrics for evaluation (if ground truth is available)
    predictions = []
    speed = []

    with torch.no_grad():
        for i_batch, (x, image_ids) in enumerate(tqdm(dataloader)):
            x = x.to(device)

            # Make prediction with timing
            t0 = time.time()
            batch_preds, batch_confs, perm_preds = test_generate(
                model, x, tokenizer, max_len=CFG.generation_steps, top_k=0, top_p=1
            )
            batch_time = time.time() - t0
            speed.append(batch_time / x.size(0))  # Time per image

            # Process predictions
            vertex_coords, confs = postprocess(batch_preds, batch_confs, tokenizer)

            # Format coordinates for polygon creation
            coords = []
            for i in range(len(vertex_coords)):
                if vertex_coords[i] is not None:
                    coord = torch.from_numpy(vertex_coords[i])
                else:
                    coord = torch.tensor([])

                padd = torch.ones((CFG.N_VERTICES - len(coord), 2)).fill_(CFG.PAD_IDX)
                coord = torch.cat([coord, padd], dim=0)
                coords.append(coord)

            # Get polygons
            batch_polygons = permutations_to_polygons(perm_preds, coords, out="torch")

            # Create masks and save predictions
            B, C, H, W = x.shape
            polygons_mask = np.zeros((B, 1, H, W))

            for b in range(len(batch_polygons)):
                # Create mask from polygons
                for c in range(len(batch_polygons[b])):
                    poly = batch_polygons[b][c]
                    poly = poly[poly[:, 0] != CFG.PAD_IDX]
                    cnt = np.flip(np.int32(poly.cpu()), 1)
                    if len(cnt) > 0:
                        cv2.fillPoly(polygons_mask[b, 0], pts=[cnt], color=1.0)

                # Save individual prediction
                original_img = cv2.imread(image_paths[i_batch * args.batch_size + b])
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

                # Save visualization
                plt.figure(figsize=(12, 6))
                plt.subplot(121)
                plt.imshow(original_img)
                plt.title("Original Image")
                plt.axis("off")

                plt.subplot(122)
                plt.imshow(polygons_mask[b, 0], cmap="gray")
                plt.title("Polygon Prediction")
                plt.axis("off")

                img_save_path = os.path.join(output_path, f"{image_ids[b]}_pred.png")
                plt.savefig(img_save_path)
                plt.close()

                # Save overlay image
                overlay = original_img.copy()
                overlay_mask = (polygons_mask[b, 0] * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    overlay_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

                plt.figure()
                plt.imshow(overlay)
                plt.axis("off")
                overlay_path = os.path.join(output_path, f"{image_ids[b]}_overlay.png")
                plt.savefig(overlay_path)
                plt.close()

                # Save COCO-style annotations
                for c in range(len(batch_polygons[b])):
                    poly = batch_polygons[b][c]
                    poly = poly[poly[:, 0] != CFG.PAD_IDX]
                    poly = torch.fliplr(poly)
                    poly = poly * (
                        original_img.shape[0] / CFG.INPUT_WIDTH
                    )  # Scale back to original size
                    poly = poly.view(-1).tolist()
                    if len(poly) > 0:
                        predictions.append(single_annotation(image_ids[b], [poly]))

            # Visualize batch results
            if i_batch % 5 == 0:  # Save every 5th batch for visualization
                polygons_tensor = torch.from_numpy(polygons_mask)
                pred_grid = make_grid(polygons_tensor).permute(1, 2, 0)
                plt.figure(figsize=(10, 10))
                plt.imshow(pred_grid)
                plt.title(f"Batch {i_batch} Predictions")
                plt.axis("off")
                plt.savefig(os.path.join(output_path, f"batch_{i_batch}_grid.png"))
                plt.close()

    # Save prediction JSON
    with open(os.path.join(output_path, "predictions.json"), "w") as fp:
        json.dump(predictions, fp)

    # Save metrics
    with open(os.path.join(output_path, "metrics.txt"), "w") as ff:
        ff.write(f"Average inference time: {np.mean(speed):.4f} seconds per image\n")
        ff.write(f"Total images processed: {len(image_paths)}\n")
        ff.write(f"Total detected polygons: {len(predictions)}\n")

    print(f"Average inference time: {np.mean(speed):.4f} seconds per image")
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
