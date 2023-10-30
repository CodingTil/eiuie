from typing import Tuple, Dict
import numpy as np
import cv2
import base_model as bm
from multiprocessing import Pool, Queue, Process, cpu_count

SAVE_LOCATION: str = "data/intermediate"


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
        cv2.imwrite(file_name, image_data)


def process_and_enqueue(
    model: bm.BaseModel, image_name: str, image_path: str, queue: Queue
) -> None:
    """
    Process the image using the model and enqueue the result for writing.

    Parameters
    ----------
    model : bm.BaseModel
        Model to be used for processing.
    image_name : str
        Name of the image.
    raw_image : np.ndarray
        Image data.
    queue : Queue
        Queue to enqueue the processed image for writing.
    """
    raw_image = cv2.imread(image_path)
    processed_image = model.process_image(raw_image)
    absolute_file_name = f"{SAVE_LOCATION}/{model.name}/{image_name}.jpg"
    queue.put((absolute_file_name, processed_image))


def batch_process(model: bm.BaseModel, image_pairs: Dict[str, Tuple[str, str]]) -> None:
    """
    Batch process images using the model, and saves the results.

    Parameters
    ----------
    model : bm.BaseModel
        Model to be used for processing.
    image_pairs : Dict[str, Tuple[str, str]]
        Dictionary of image pairs, with image name as key and tuple of absolute paths as value.
    """
    # Create a queue for writing
    write_queue = Queue()

    # Start the writing process
    writer_process = Process(target=write_to_file, args=(write_queue,))
    writer_process.start()

    # Create a pool for parallel processing
    with Pool(cpu_count() - 1) as pool:
        for image_name, (raw_image, _) in image_pairs.items():
            pool.apply_async(
                process_and_enqueue, (model, image_name, raw_image, write_queue)
            )

        # Close the pool and wait for the tasks to complete
        pool.close()
        pool.join()

    # Notify the writer process to stop
    write_queue.put(("STOP", None))
    writer_process.join()
