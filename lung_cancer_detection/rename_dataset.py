import argparse
import os
import shutil

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def normalize_class(folder_name):
    name = folder_name.lower()
    if "normal" in name:
        return "lung_n"
    if "adenocarcinoma" in name:
        return "lung_aca"
    if "squamous" in name:
        return "lung_scc"
    return None


def copy_split(src_split, dst_split):
    os.makedirs(dst_split, exist_ok=True)
    counts = {"lung_n": 0, "lung_aca": 0, "lung_scc": 0}
    for folder in os.listdir(src_split):
        src_folder = os.path.join(src_split, folder)
        if not os.path.isdir(src_folder):
            continue
        mapped = normalize_class(folder)
        if not mapped:
            print(f"Skipping class: {folder}")
            continue
        dst_class_dir = os.path.join(dst_split, mapped)
        os.makedirs(dst_class_dir, exist_ok=True)
        for root, _, files in os.walk(src_folder):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in VALID_EXTS:
                    continue
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_class_dir, file)
                if os.path.exists(dst_file):
                    base, ext = os.path.splitext(file)
                    dst_file = os.path.join(dst_class_dir, f"{base}_{os.path.getmtime(src_file):.0f}{ext}")
                shutil.copy2(src_file, dst_file)
                counts[mapped] += 1
    return counts


def main(args):
    src_root = args.src
    dst_root = args.dst

    for split in ["train", "valid", "test"]:
        src_split = os.path.join(src_root, split)
        if not os.path.isdir(src_split):
            print(f"Missing split: {src_split}")
            continue
        dst_split = os.path.join(dst_root, split)
        counts = copy_split(src_split, dst_split)
        print(f"{split}: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize dataset classes for lung cancer model.")
    parser.add_argument("--src", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset")))
    parser.add_argument("--dst", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset_std")))
    args = parser.parse_args()
    main(args)
