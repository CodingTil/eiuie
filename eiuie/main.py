import argparse

import cv2

import download as download_module
import image_getter as ig
import batch_process as bp
import base_model as bm
import unsharp_masking
import retinex
import homomorphic_filtering


def main():
    parser = argparse.ArgumentParser(description="Entry point for the eiuie CLI.")

    parser.add_argument(
        "command",
        type=str,
        choices=["download", "single", "batch_process"],
        help="Command to run",
    )

    # --method=xyz
    parser.add_argument(
        "--method",
        type=str,
        default="unsharp_masking",
        choices=["unsharp_masking", "retinex", "homomorphic_filtering"],
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
        case _:
            raise ValueError(f"Unknown method: {args.method}")

    match args.command:
        case "download":
            download_module.main()
        case "single":
            image = cv2.imread(args.file)
            processed_image = method.process_image(image)
            # show image
            cv2.imshow("image", processed_image)
            cv2.waitKey()
        case "batch_process":
            images = ig.get_all_image_pairs()
            print(f"Processing {len(images)} images...")
            print(images)
            bp.batch_process(method, images)
        case _:
            raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
