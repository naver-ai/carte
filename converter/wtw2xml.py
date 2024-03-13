"""
CARTE
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
"""
import argparse
import os
import shutil
import xml.etree.ElementTree as et
from copy import deepcopy
from glob import glob
from xml.dom import minidom

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Convert from WTW formatted data to xml data.")
parser.add_argument("-s", "--source_path", type=str, default="/data/db/table/WTW")
parser.add_argument("-t", "--target_path", type=str, default="/data/db/table/WTWCell")
args = parser.parse_args()

src_dir = args.source_path
target_dir = args.target_path

# original xml
src_img_dir = "images/"
src_gt_dir = "xml/"
data_types = ["train", "test"]

# revised xml
# src_gt_dir = "test-xml-revise/"
# data_types = ["test"]

gt_ext = ".xml"

for data_type in data_types:
    target_type_dir = os.path.join(target_dir, data_type)

    # mkdir
    if os.path.exists(target_type_dir):
        print(f"Target path {target_type_dir} already exists.")
        exit(1)
        # shutil.rmtree(target_type_dir)
    os.makedirs(target_type_dir, exist_ok=True)

    # img_files = glob(os.path.join(src_img_dir, '**/*.*'), recursive=True)
    gt_files = glob(
        os.path.join(src_dir, data_type, src_gt_dir, "**/*.*"), recursive=True
    )

    # convert
    table_template = {"coords": [], "cells": [], "max_row": -1, "max_col": -1}
    for gt_filename in tqdm(gt_files):
        gt = {"image_url": "", "tables": []}
        tree = et.parse(gt_filename)
        root = tree.getroot()
        table = deepcopy(table_template)
        for info in root:
            if info.tag == "filename":
                gt["image_url"] = info.text
            if info.tag == "object":
                for obj in info:
                    if obj.tag == "bndbox":
                        tid = -1
                        x1, x2, x3, x4 = 0, 0, 0, 0
                        y1, y2, y3, y4 = 0, 0, 0, 0
                        start_col, start_row = 0, 0
                        end_col, end_row = 0, 0
                        for bbox in obj:
                            if bbox.tag == "x1":
                                x1 = float(bbox.text)
                            elif bbox.tag == "x2":
                                x2 = float(bbox.text)
                            elif bbox.tag == "x3":
                                x3 = float(bbox.text)
                            elif bbox.tag == "x4":
                                x4 = float(bbox.text)
                            elif bbox.tag == "y1":
                                y1 = float(bbox.text)
                            elif bbox.tag == "y2":
                                y2 = float(bbox.text)
                            elif bbox.tag == "y3":
                                y3 = float(bbox.text)
                            elif bbox.tag == "y4":
                                y4 = float(bbox.text)
                            elif bbox.tag == "startcol":
                                start_col = int(bbox.text)
                            elif bbox.tag == "startrow":
                                start_row = int(bbox.text)
                            elif bbox.tag == "endcol":
                                end_col = int(bbox.text)
                            elif bbox.tag == "endrow":
                                end_row = int(bbox.text)
                            elif bbox.tag == "tableid":
                                tid = int(bbox.text)
                        if tid != -1:
                            while tid >= len(gt["tables"]):
                                gt["tables"].append(table)
                                table = deepcopy(table_template)
                            cell = {
                                "start-row": start_row,
                                "end-row": end_row,
                                "start-col": start_col,
                                "end-col": end_col,
                                "coords": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                            }
                            gt["tables"][tid]["cells"].append(cell)
                            gt["tables"][tid]["max_row"] = max(
                                end_row, gt["tables"][tid]["max_row"]
                            )
                            gt["tables"][tid]["max_col"] = max(
                                end_col, gt["tables"][tid]["max_col"]
                            )
        # quads = [[], [], [], []]
        for table in gt["tables"]:
            table["coords"] = [[-1, -1]] * 4
        #     max_row, max_col = table['max_row'], table['max_col']
        #     for cell in table['cells']:
        #         if cell['start-row'] == 0 and cell['start-col'] == 0:
        #             quads[0] = cell['coords'][0]
        #         elif cell['start-row'] == 0 and cell['end-col'] == max_col:
        #             quads[1] = cell['coords'][1]
        #         elif cell['end-row'] == max_row and cell['end-col'] == max_col:
        #             quads[2] = cell['coords'][2]
        #         elif cell['end-row'] == max_row and cell['start-col'] == 0:
        #             quads[3] = cell['coords'][3]
        #     table['coords'] = quads

        image_url = gt["image_url"]

        # copy img file
        shutil.copy(
            os.path.join(src_dir, data_type, src_img_dir, image_url), target_type_dir
        )

        # create xml file
        filename = os.path.splitext(os.path.basename(image_url))[0]
        xml_filename = filename + ".xml"
        root = minidom.Document()
        xml = root.createElement("document")
        xml.setAttribute("filename", os.path.basename(image_url))

        for table in gt["tables"]:
            quad = table["coords"]
            table_elem = root.createElement("table")
            table_coords = root.createElement("Coords")
            table_coords.setAttribute(
                "points",
                f"{int(quad[0][0])},{int(quad[0][1])} {int(quad[1][0])},{int(quad[1][1])} {int(quad[2][0])},{int(quad[2][1])} {int(quad[3][0])},{int(quad[3][1])}",
            )
            table_elem.appendChild(table_coords)

            for cell in table["cells"]:
                cell_elem = root.createElement("cell")
                # cell id
                cell_elem.setAttribute("start-row", f"{cell['start-row']}")
                cell_elem.setAttribute("end-row", f"{cell['end-row']}")
                cell_elem.setAttribute("start-col", f"{cell['start-col']}")
                cell_elem.setAttribute("end-col", f"{cell['end-col']}")
                # cell coordinates
                cell_coords = root.createElement("Coords")
                x1, y1 = cell["coords"][0]
                x2, y2 = cell["coords"][1]
                x3, y3 = cell["coords"][2]
                x4, y4 = cell["coords"][3]
                cell_coords.setAttribute(
                    "points", f"{x1},{y1} {x2},{y2} {x3},{y3} {x4},{y4}"
                )
                # cell attributes
                cell_lines = root.createElement("Lines")
                cell_lines.setAttribute("top", f"{1}")
                cell_lines.setAttribute("bottom", f"{1}")
                cell_lines.setAttribute("left", f"{1}")
                cell_lines.setAttribute("right", f"{1}")

                cell_elem.appendChild(cell_coords)
                cell_elem.appendChild(cell_lines)
                table_elem.appendChild(cell_elem)
            xml.appendChild(table_elem)
        root.appendChild(xml)

        xml_string = root.toprettyxml(indent="\t")
        with open(os.path.join(target_type_dir, xml_filename), "w") as fp:
            fp.write(xml_string)
