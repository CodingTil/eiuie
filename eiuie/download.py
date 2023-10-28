import os
import requests
from pathlib import Path
import rawpy
from PIL import Image
from tqdm import tqdm
from joblib import Parallel


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def dng_to_jpg(img_path: str):
    raw = rawpy.imread(img_path)
    rgb = raw.postprocess(use_camera_wb=True)
    new_name = img_path[:-3] + "jpg"
    Image.fromarray(rgb).save(new_name, optimize=True)
    os.remove(img_path)


def tif_to_jpg(img_path: str):
    img = Image.open(img_path)
    img.save(img_path[:-3] + "jpg")
    os.remove(img_path)


def download_img(img_name: str, data_dir: Path):
    try:
        dng_url = f"https://data.csail.mit.edu/graphics/fivek/img/dng/{img_name}.dng"
        dng_file = requests.get(dng_url, allow_redirects=True)
        dng_path = data_dir / "raw" / (img_name + ".dng")
        open(dng_path, "wb").write(dng_file.content)
        dng_to_jpg(str(dng_path))

        tif_base = "https://data.csail.mit.edu/graphics/fivek/img/tiff16"
        for expert in ["a", "b", "c", "d", "e"]:
            url = f"{tif_base}_{expert}/{img_name}.tif"
            path = data_dir / expert / (img_name + ".tif")
            tif_file = requests.get(url, allow_redirects=True)
            open(path, "wb").write(tif_file.content)
            tif_to_jpg(str(path))

    except Exception as e:
        print(img_name, e)


def download_dataset(store_dir: Path, n_jobs: int = 8):
    # * Create folders
    dng_dir = store_dir / "raw"
    tif_dirs = [store_dir / s for s in ["a", "b", "c", "d", "e"]]

    dng_dir.mkdir(parents=True, exist_ok=True)
    for path in tif_dirs:
        path.mkdir(parents=True, exist_ok=True)

    # * Get image info
    f1 = requests.get(
        "https://data.csail.mit.edu/graphics/fivek/legal/filesAdobe.txt"
    ).text.split("\n")
    f2 = requests.get(
        "https://data.csail.mit.edu/graphics/fivek/legal/filesAdobeMIT.txt"
    ).text.split("\n")
    names = [x for x in set(f1 + f2) if x != ""]

    # * Download imgs
    ProgressParallel(n_jobs=n_jobs, total=len(names))(
        download_img(name, store_dir) for name in names
    )


def main():
    store_dir = Path("data/fivek")
    download_dataset(store_dir, os.cpu_count() or 1)


if __name__ == "__main__":
    main()
