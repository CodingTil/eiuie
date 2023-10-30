from typing import Dict, List
from multiprocessing import Pool, Queue, Process, Manager, cpu_count
import glob
import os

import cv2

import base_model as bm
import unsharp_masking
import retinex
import homomorphic_filtering

SAVE_LOCATION: str = "data/intermediate_images"


def write_to_file(queue: Queue) -> None:
    """
    Continuously write images from the queue to the file system.

    Parameters
    ----------
    queue : Queue
        Queue containing tuples of file name and image data.
    """
    while True:
        file_name, image_data = queue.get()
        if file_name == "STOP":
            break
        print(f"Writing {file_name}")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        cv2.imwrite(file_name, image_data)


def process_and_enqueue(
    models: List[bm.BaseModel], image_name: str, image_path: str, queue: Queue
) -> None:
    """
    Process the image using the models and enqueue the result for writing.

    Parameters
    ----------
    models : List[bm.BaseModel]
        List of models to be used for processing.
    image_name : str
        Name of the image.
    raw_image : np.ndarray
        Image data.
    queue : Queue
        Queue to enqueue the processed image for writing.
    """
    image = cv2.imread(image_path)
    for model in models:
        print(f"Processing {image_name} with {model.name}")
        processed_image = model.process_image(image)
        absolute_file_name = f"{SAVE_LOCATION}/{model.name}/{image_name}.png"
        absolute_file_name = os.path.abspath(absolute_file_name)
        queue.put((absolute_file_name, processed_image))


def batch_process(models: List[bm.BaseModel], images: Dict[str, str]) -> None:
    """
    Batch process images using the model, and saves the results.

    Parameters
    ----------
    models : List[bm.BaseModel]
        List of models to be used for processing.
    images : Dict[str, str]
        Dictionary containing the image name and absolute path.
    """
    # Create a managed queue for writing
    with Manager() as manager:
        write_queue = manager.Queue()

        # Start the writing process
        writer_process = Process(target=write_to_file, args=(write_queue,))
        writer_process.start()

        print(f"Processing {len(images)} images")

        # Create a pool for parallel processing
        async_results = []  # Collect all the AsyncResult objects here
        with Pool(cpu_count() - 1) as pool:
            for image_name, image_path in images.items():
                res = pool.apply_async(
                    process_and_enqueue,
                    args=(models, image_name, image_path, write_queue),
                )
                async_results.append(res)

            print("Waiting for all tasks to complete")
            # Ensure all tasks have finished
            for result in async_results:
                result.wait()

            # Close the pool and wait for the tasks to complete
            pool.close()
            pool.join()

        # Notify the writer process to stop
        write_queue.put(("STOP", None))
        writer_process.join()


def batch_process_dataset() -> None:
    glob_pattern = "data/lol_dataset/*/low/*.png"
    images = glob.glob(glob_pattern)
    images_dict = {
        image.split("/")[-1].split(".")[0]: os.path.abspath(image) for image in images
    }
    print(f"Found {len(images_dict)} images")
    models = [
        unsharp_masking.UnsharpMasking(),
        homomorphic_filtering.HomomorphicFiltering(),
        retinex.Retinex(),
    ]
    batch_process(models, images_dict)
