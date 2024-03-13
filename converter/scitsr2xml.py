"""
CARTE
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from argparse import ArgumentError, ArgumentParser
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
from xml.dom import minidom

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(prog="SciTSR converter",
                                 description="Convert from SciTSR formatted data to xml data. The converted XML files will be saved in (root_path)/(test/train)/xml/.")
parser.add_argument("-s", "--source_path", type=str, default="/data/db/table/scitsr/test")
args = parser.parse_args()


# default dirpath
structure_dirpath = os.path.join(args.source_path, "structure")
chunk_dirpath = os.path.join(args.source_path, "chunk")
output_dirpath = os.path.join(args.source_path, "xml")

# original image size with 150 dpi applied
X_SCALE = 1241
Y_SCALE = 1754
IMAGE_SIZE_ORG = (595, 841)

TwoPoints = Tuple[float, float, float, float]

Quad = Tuple[
    Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]
]


@dataclass
class Cell:
    id: int
    text_org: str
    text_remain: str
    start_row: str
    end_row: str
    start_col: str
    end_col: str
    chunk_ids: List[int] = field(default_factory=list)
    quads: Optional[List[Quad]] = field(default_factory=list, repr=False)
    quad: Optional[Quad] = field(default=None, repr=False)


@dataclass
class Chunk:
    id: int
    text_org: str
    quad: Optional[Quad] = field(default=None, repr=False)


class ArgumentParser(ArgumentParser):
    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, "%s: error: %s\n" % (self.prog, message))

def maximum_coords(quads: List[Quad]) -> Quad:
    arr = np.array(quads).round(0)
    assert arr.shape[1:] == (4, 2), arr

    min_x = arr[:, :, 0].min().tolist()
    max_x = arr[:, :, 0].max().tolist()
    min_y = arr[:, :, 1].min().tolist()
    max_y = arr[:, :, 1].max().tolist()
    # counterclock-wise from bottom left point
    return ((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y))


def convert_pos(pos: TwoPoints) -> Quad:
    """Convert two-points representation to quads"""
    x_min, x_max, y_min, y_max = np.array(pos).clip(0).tolist()

    # counterclock-wise
    quad = ((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max))
    return quad


def translate_coords(
    quad: Quad, imsize_org: Tuple[float, float], imsize_new: Tuple[float, float]
) -> Quad:
    scale = np.array(imsize_new) / np.array(imsize_org)
    new_quad = (np.array(quad) * scale).tolist()
    return new_quad


def load_structure(structure_filepath: str) -> List[Cell]:
    structure_obj = json.loads(Path(structure_filepath).read_text())["cells"]
    structure = []
    for i_c, c in enumerate(structure_obj):
        if c["content"]:
            cell_inst = Cell(
                id=int(c["id"]),
                text_org="".join("".join(c["content"]).split()),
                text_remain="".join("".join(c["content"]).split()),
                start_row=str(c["start_row"]),
                end_row=str(c["end_row"]),
                start_col=str(c["start_col"]),
                end_col=str(c["end_col"]),
            )
            structure.append(cell_inst)
    structure = sorted(structure, key=lambda x: x.id)
    return structure


def load_chunk(chunk_filepath: str) -> List[Chunk]:
    chunk_obj = json.loads(Path(chunk_filepath).read_text())["chunks"]
    chunk = []
    for i_c, c in enumerate(chunk_obj):
        if c["text"]:
            chunk_inst = Chunk(
                id=i_c,
                text_org="".join(c["text"].split()),
                quad=convert_pos(c["pos"]),
            )
            chunk.append(chunk_inst)
    return chunk


def align(structure: List[Cell], chunk: List[Chunk]):
    aligned_cell = set()
    aligned_chunk = set()
    for i_ce, ce in enumerate(structure):
        for i_ch, ch in enumerate(chunk):
            if (i_ce == len(structure) - 1) and (i_ch == len(chunk) - 1):
                min_span = min(len(ce.text_remain), len(ch.text_org))
                text_ce = ce.text_remain[:min_span]
                text_ch = ch.text_org[:min_span]
            else:
                text_ce = ce.text_remain
                text_ch = ch.text_org

            if i_ch not in aligned_chunk:
                common_prefix = os.path.commonprefix([text_ce, text_ch])
                if common_prefix:
                    ce.chunk_ids.append(i_ch)
                    ce.quads.append(ch.quad)
                    ce.text_remain = ce.text_remain[len(common_prefix) :]
                    aligned_cell.add(i_ce)
                    aligned_chunk.add(i_ch)

    num_missing_cell = len(structure) - len(aligned_cell)

    # remove remaining cells
    structure = [ce for ce in structure if ce.quads]

    return structure, num_missing_cell, aligned_cell, aligned_chunk


def convert_to_xml(
    structure: List[Cell],
    filename: str,
    image_size: Tuple[float, float],
) -> ET:
    root = ET.Element("document", {"filename": filename})
    ET.SubElement(root, "image", attrib={"size": f"{image_size[0]},{image_size[1]}"})
    table = ET.SubElement(root, "table")

    # table quad
    quads = []
    for ce in structure:
        if ce.quads:
            quads.extend(ce.quads)
    table_quad = maximum_coords(quads)
    # translate
    table_quad = translate_coords(table_quad, IMAGE_SIZE_ORG, image_size)

    ET.SubElement(
        table,
        "Coords",
        attrib={
            "points": (
                f"{table_quad[3][0]},{image_size[1]-table_quad[3][1]} "
                f"{table_quad[2][0]},{image_size[1]-table_quad[2][1]} "
                f"{table_quad[1][0]},{image_size[1]-table_quad[1][1]} "
                f"{table_quad[0][0]},{image_size[1]-table_quad[0][1]}"
            )
        },
    )

    for i_ce, ce in enumerate(structure):
        c = ET.SubElement(
            table,
            "cell",
            attrib={
                "start-row": ce.start_row,
                "start-col": ce.start_col,
                "end-row": ce.end_row,
                "end-col": ce.end_col,
            },
        )

        ET.SubElement(c, "content", attrib={"text": ce.text_org})

        quad = maximum_coords(ce.quads)
        # translate
        quad = translate_coords(quad, IMAGE_SIZE_ORG, image_size)
        ET.SubElement(
            c,
            "Coords",
            attrib={
                "points": (
                    f"{quad[3][0]},{image_size[1]-quad[3][1]} "
                    f"{quad[2][0]},{image_size[1]-quad[2][1]} "
                    f"{quad[1][0]},{image_size[1]-quad[1][1]} "
                    f"{quad[0][0]},{image_size[1]-quad[0][1]}"
                )
            },
        )
    return root


def write_xml(write_filepath: str, root: ET):
    # write file
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(
        indent="    ", encoding="UTF-8"
    )
    with open(write_filepath, "wb") as wf:
        wf.write(xmlstr)


def run(args: dict):
    structure_dir = structure_dirpath
    print(f"json dirpath: {Path(structure_dir).resolve().as_posix()}")
    chunk_dir = chunk_dirpath
    print(f"chunk dirpath: {Path(chunk_dir).resolve().as_posix()}")

    structure_fpath_list = list(Path(structure_dir).glob("*.json"))
    print(f"num files to convert: {len(structure_fpath_list)}")

    if structure_fpath_list:
        wfpath = Path(output_dirpath).resolve()
        wfpath.mkdir(parents=True, exist_ok=True)
        print(f"save xml files to: {wfpath.as_posix()}")

    cnt = 0
    error_cnt = []
    for target_fpath in tqdm(structure_fpath_list):
        # load
        structure = load_structure(target_fpath)
        chunk_filepath = Path(chunk_dir, target_fpath.stem + ".chunk").as_posix()
        chunk = load_chunk(chunk_filepath)

        # align
        structure, num_missing_cell, _, _ = align(structure, chunk)

        # convert
        try:
            et = convert_to_xml(
                structure,
                Path(target_fpath).stem + ".png",
                (X_SCALE, Y_SCALE)
            )
        except:
            raise Exception(f"error on {target_fpath}")

        # write
        write_xml(Path(wfpath, Path(target_fpath).stem + ".xml"), et)

        error_cnt.append(num_missing_cell)
        cnt += 1

    print("num err:", Counter(error_cnt))
    # print(structure_fpath_list[error_cnt.index(max(error_cnt))])
    print(f"Done with {cnt} files.")


if __name__ == "__main__":
    run(args)