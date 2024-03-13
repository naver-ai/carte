"""
CARTE
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
"""
import argparse
import glob
import os
from multiprocessing import Pool
import json

from tqdm import tqdm

from carte import Statistics, eval_carte

parser = argparse.ArgumentParser(description="CARTE argument prser")
parser.add_argument(
    "-g", "--gt_path", default="./sample_gt/", help="Path of the ground truth files."
)
parser.add_argument(
    "-p",
    "--pred_path",
    default="./sample_result/",
    help="Path of your prediction files.",
)
parser.add_argument(
    "-n",
    "--no_show_progress",
    default=False,
    action="store_true",
    help="Whether to show evaluation progress.",
)
parser.add_argument(
    "-s",
    "--show_single_result",
    default=False,
    action="store_true",
    help="Show stastics of each file (no multi-processing).",
)
parser.add_argument(
    "-t",
    "--matching_text",
    default=False,
    action="store_true",
    help="Matching with text LCS, not based on cell or content position.",
)
parser.add_argument(
    "-mp", "--num_mp", type=int, default=10, help="Number of multi-processings"
)
parser.add_argument(
    "-save",
    "--save_single_stat",
    default=False,
    action="store_true",
    help="Save statistics of each individual file at the end."
)
args = parser.parse_args()

if os.path.isdir(args.gt_path):
    gt_file_lst = glob.glob(args.gt_path + "/*.xml")
else:
    gt_file_lst = [args.gt_path]

pool = Pool(processes=args.num_mp)

tot_stat = Statistics()
if not args.no_show_progress:
    pbar = tqdm(total=len(gt_file_lst))

mp_args = [
    (gt_file, args.pred_path, args.matching_text, args.show_single_result)
    for gt_file in gt_file_lst
]
single_stat = {}
for stat in pool.imap_unordered(eval_carte, mp_args):
    tot_stat.update(stat)
    if not args.no_show_progress:
        pbar.update(1)
    if args.save_single_stat:
        if stat._fp + stat._fn > 0:
            single_stat[stat._filename] ={
                'gt': stat._gt,
                'gt': stat._gt,
                'tp': stat._tp,
                'fp': stat._fp,
                'fn': stat._fn,
            }

print("---------------------")
tot_stat.print_detail()

if args.save_single_stat:
    with open('output_single_stat.json', 'w') as json_file:
        json.dump(single_stat, json_file)