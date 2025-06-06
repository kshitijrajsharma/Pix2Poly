import argparse
import os
import time

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from config import CFG
from models.model import Decoder, Encoder, EncoderDecoder
from tokenizer import Tokenizer
from utils import load_checkpoint, permutations_to_polygons, postprocess, test_generate


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", required=True, help="Path to input image")
    parser.add_argument(
        "-c", "--checkpoint_path", required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="output.png",
        help="Path to save prediction visualization",
    )
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CFG.DEVICE = device

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

    # Load and transform image
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transforms(image=image)
    x = transformed["image"].unsqueeze(0).to(device)  # Add batch dimension

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

    # Make prediction
    with torch.no_grad():
        t0 = time.time()
        batch_preds, batch_confs, perm_preds = test_generate(
            model, x, tokenizer, max_len=CFG.generation_steps, top_k=0, top_p=1
        )
        inference_time = time.time() - t0
        print(f"Inference time: {inference_time:.4f} seconds")

        # Process predictions
        vertex_coords, confs = postprocess(batch_preds, batch_confs, tokenizer)

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

        # Create mask from polygons
        polygons_mask = np.zeros((1, 1, CFG.INPUT_HEIGHT, CFG.INPUT_WIDTH))
        for b in range(len(batch_polygons)):
            for c in range(len(batch_polygons[b])):
                poly = batch_polygons[b][c]
                poly = poly[poly[:, 0] != CFG.PAD_IDX]
                cnt = np.flip(np.int32(poly.cpu()), 1)
                if len(cnt) > 0:
                    cv2.fillPoly(polygons_mask[b, 0], pts=[cnt], color=1.0)

        # Visualize results
        plt.figure(figsize=(12, 6))

        # Original image
        plt.subplot(121)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        # Prediction mask
        plt.subplot(122)
        plt.imshow(polygons_mask[0, 0], cmap="gray")
        plt.title("Polygon Prediction")
        plt.axis("off")

        # Save visualization
        plt.savefig(args.output_path)
        print(f"Prediction saved to {args.output_path}")

        # Overlay prediction on original image
        overlay = image.copy()
        overlay_mask = (polygons_mask[0, 0] * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            overlay_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        # Save overlay image
        overlay_path = os.path.splitext(args.output_path)[0] + "_overlay.png"
        plt.figure()
        plt.imshow(overlay)
        plt.axis("off")
        plt.savefig(overlay_path)
        print(f"Overlay saved to {overlay_path}")

        return batch_polygons


if __name__ == "__main__":
    main()
