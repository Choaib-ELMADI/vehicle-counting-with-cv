import os
import argparse
import time
import ast
import torch
import gc

from counting.run_count import run
from counting.count import args

import pandas as pd


def duplicate_video_10_times(input_path, output_path):
    with open("list.txt", "w") as f:
        for _ in range(10):
            f.write(f"file '{input_path}'\n")
    os.system(f"ffmpeg -y -f concat -safe 0 -i list.txt -c copy {output_path}")
    os.remove("list.txt")


def main(vid_strides):
    args.counting_approach = "tracking_with_line_vicinity"
    args.save = False
    args.verbose = False
    args.use_mask = False
    args.save_csv_count = False
    args.tracking_method = "ocsort"

    folder = os.path.join(os.getcwd(), "dataset")
    video = "kech1.mp4"
    original_path = os.path.join(folder, video)
    duplicated_path = os.path.join(folder, f"duplicated_{video}")

    duplicate_video_10_times(original_path, duplicated_path)
    args.source = duplicated_path

    line = ast.literal_eval(
        pd.read_csv(os.path.join(folder, "actual_counts.csv"))["line_of_counting"][0]
    )
    args.line_point11, args.line_point12 = line
    line_vicinities = [2, 1.5, 1, 0.5]

    for stride in vid_strides:
        args.vid_stride = stride
        for lv in line_vicinities:
            args.line_vicinity = lv
            run(args)
            torch.cuda.empty_cache()
            gc.collect()

    os.remove(duplicated_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_strides", type=int, nargs="+", required=True)
    args_parser = parser.parse_args()
    main(args_parser.vid_strides)
