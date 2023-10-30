import argparse

import cv2

import batch_process as bp
import base_model as bm
import consolidate_dataset as cd
import unsharp_masking
import retinex
import homomorphic_filtering
import fusion_model


def main():
    parser = argparse.ArgumentParser(description="Entry point for the eiuie CLI.")

    parser.add_argument(
        "command",
        type=str,
        choices=["single", "batch_process", "prepare_dataset", "train"],
        help="Command to run",
    )

    # --method=xyz
    parser.add_argument(
        "--method",
        type=str,
        default="unsharp_masking",
        choices=["unsharp_masking", "retinex", "homomorphic_filtering", "fusion_model"],
        help="Filter method to use",
    )

    # --file=xyz
    parser.add_argument(
        "--file",
        type=str,
        help="Path to image file to process",
    )

    args = parser.parse_args()

    method: bm.BaseModel
    match args.method:
        case "unsharp_masking":
            method = unsharp_masking.UnsharpMasking()
        case "retinex":
            method = retinex.Retinex()
        case "homomorphic_filtering":
            method = homomorphic_filtering.HomomorphicFiltering()
        case "fusion_model":
            method = fusion_model.FusionModel()
        case _:
            raise ValueError(f"Unknown method: {args.method}")

    match args.command:
        case "single":
            image = cv2.imread(args.file)
            processed_image = method.process_image(image)
            # show image
            cv2.imshow("image", processed_image)
            cv2.waitKey()
        case "batch_process":
            bp.batch_process_dataset()
        case "prepare_dataset":
            cd.prepare_dataset()
        case "train":
            method = fusion_model.FusionModel()
            method.train_model()
        case _:
            raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
