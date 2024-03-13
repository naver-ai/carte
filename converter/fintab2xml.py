"""
CARTE
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
# from pdf2image import convert_from_path # needed for converting pdf
from PyPDF2 import PdfReader
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Convert from FinTabNet formatted data to xml data. The converted XML files will be saved in (root_path)/xml/.")
parser.add_argument("-s", "--source_path", type=str, default="/data/db/table/fintabnet")
args = parser.parse_args()


MAX_TABLE_LENGTH = 10000
FAILED_LIST = [
    "BSX/2007/page_92.pdf",
    "AIG/2018/page_68.pdf",
    "AIG/2017/page_62.pdf",
    "WHR/2018/page_29.pdf",
]


def get_data_types(jsonl_path):
    filestem = jsonl_path.stem
    data_type, data_split = filestem.split("_")[-2:]
    return data_type, data_split


def get_pdf_size(pdfpath):
    pdf_page = PdfReader(open(pdfpath, "rb")).pages[0]
    pdf_shape = pdf_page.mediabox
    pdf_height = int(pdf_shape[3] - pdf_shape[1])
    pdf_width = int(pdf_shape[2] - pdf_shape[0])

    return pdf_width, pdf_height


def convert_coord_lb_to_lt(pdf_width, pdf_height, lb_l, lb_b, lb_r, lb_t):
    bb_l = lb_l
    bb_t = pdf_height - lb_t
    bb_r = lb_r
    bb_b = pdf_height - lb_b

    return bb_l, bb_t, bb_r, bb_b


def merge_resplit(tags):
    tag_str = "".join(tags)
    split_tags = re.finditer("<.*?>", tag_str)

    return [each.group() for each in split_tags]


def extract_span(spanstr):
    rowspan = 1
    colspan = 1
    if "rowspan" in spanstr:
        rowspan = int(spanstr.split('rowspan="')[1].split('"')[0])
    if "colspan" in spanstr:
        colspan = int(spanstr.split('colspan="')[1].split('"')[0])

    return rowspan, colspan


def write_to_xml(rootpath, data_type, data_split, table_filename, xml_format):
    table_xmlname = table_filename.replace("pdf", "xml").replace("/", "__")
    xml_path = os.path.join(rootpath, "xml", data_type, data_split, table_xmlname)
    os.makedirs(os.path.dirname(xml_path), exist_ok=True)
    with open(xml_path, "w") as fout:
        fout.write(xml_format)
    return


def tag_to_tsize(html_tag):
    td_count_arr = np.zeros(MAX_TABLE_LENGTH)
    tr_count = 0
    for each in html_tag:
        if each.startswith("<tr"):
            tr_count += 1
        if each.startswith("<td"):
            rowspan, colspan = extract_span(each)
            for tridx in range(rowspan):
                td_count_arr[tr_count + tridx] += colspan

    td_count = int(td_count_arr.max())

    return tr_count, td_count


def cell_format(st_r, st_c, end_r, end_c, x1, y1, x2, y2, x3, y3, x4, y4):
    x1, y1, x2, y2, x3, y3, x4, y4 = [
        int(each) for each in [x1, y1, x2, y2, x3, y3, x4, y4]
    ]
    cell_str = f"""    <cell start-row="{st_r}" start-col="{st_c}" end-row="{end_r}" end-col="{end_c}">\n      <Coords points="{x1},{y1} {x2},{y2} {x3},{y3} {x4},{y4}" />\n    </cell>\n"""

    return cell_str


def table_format(x1, y1, x2, y2, x3, y3, x4, y4, cell_str):
    x1, y1, x2, y2, x3, y3, x4, y4 = [
        int(each) for each in [x1, y1, x2, y2, x3, y3, x4, y4]
    ]
    table_str = f"""  <table>\n    <Coords points="{x1},{y1} {x2},{y2} {x3},{y3} {x4},{y4}" />\n{cell_str}  </table>\n"""

    return table_str


def xml_format(filename, table_str):
    xml_str = f"""<?xml version="1.0" encoding="UTF-8"?>\n<document filename="{filename}">\n{table_str}\n</document>"""

    return xml_str


def process_jsonl(root_path, inputjsonl):
    annotations = {"test": dict(), "train": dict(), "val": dict()}

    print("processing: ", inputjsonl)
    with open(inputjsonl, "r") as fin:
        for line in tqdm(fin):
            sample = json.loads(line)

            tags = sample["html"]["structure"]["tokens"]

            tags = merge_resplit(tags)
            tr_count, td_count = tag_to_tsize(tags)

            table_traverse = np.full((tr_count, td_count), False)

            table_filename = sample["filename"]
            if table_filename in FAILED_LIST:
                continue
            pdf_width, pdf_height = get_pdf_size(
                os.path.join(root_path, "pdf", table_filename)
            )

            data_split = sample["split"]

            r_idx = -1
            c_idx = -1
            cells_str = ""
            data_idx = 0

            for t, token in enumerate(tags):
                if token == "<tr>":
                    r_idx += 1
                    c_idx = -1
                    continue
                if token.startswith("<td"):
                    c_idx += 1
                    while table_traverse[r_idx][c_idx]:
                        c_idx += 1
                    rowspan, colspan = extract_span(token)

                    for rr in range(r_idx, r_idx + rowspan):
                        for cc in range(c_idx, c_idx + colspan):
                            table_traverse[rr][cc] = True

                    if "bbox" in sample["html"]["cells"][data_idx]:
                        lb_l, lb_b, lb_r, lb_t = sample["html"]["cells"][data_idx][
                            "bbox"
                        ]
                        bb_l, bb_t, bb_r, bb_b = convert_coord_lb_to_lt(
                            pdf_width, pdf_height, lb_l, lb_b, lb_r, lb_t
                        )
                        cells_str += cell_format(
                            r_idx,
                            c_idx,
                            r_idx + rowspan - 1,
                            c_idx + colspan - 1,
                            bb_l,
                            bb_t,
                            bb_r,
                            bb_t,
                            bb_r,
                            bb_b,
                            bb_l,
                            bb_b,
                        )

                    data_idx += 1

            t_lb_l, t_lb_b, t_lb_r, t_lb_t = sample["bbox"]
            t_bb_l, t_bb_t, t_bb_r, t_bb_b = convert_coord_lb_to_lt(
                pdf_width, pdf_height, t_lb_l, t_lb_b, t_lb_r, t_lb_t
            )

            if table_filename in annotations[data_split]:
                annotations[data_split][sample["filename"]] += table_format(
                    t_bb_l,
                    t_bb_t,
                    t_bb_r,
                    t_bb_t,
                    t_bb_r,
                    t_bb_b,
                    t_bb_l,
                    t_bb_b,
                    cells_str,
                )
            else:
                annotations[data_split][sample["filename"]] = table_format(
                    t_bb_l,
                    t_bb_t,
                    t_bb_r,
                    t_bb_t,
                    t_bb_r,
                    t_bb_b,
                    t_bb_l,
                    t_bb_b,
                    cells_str,
                )

    data_type, _ = get_data_types(Path(inputjsonl))

    for split, annot in annotations.items():
        for table_filename, table_str in annot.items():
            write_to_xml(
                root_path,
                data_type,
                split,
                table_filename,
                xml_format(table_filename, table_str),
            )


def main(fintabnet_rootpath="examples"):
    path = Path(fintabnet_rootpath)
    jsonlfiles = path.glob("*.jsonl")

    for eachjsonlfile in jsonlfiles:
        process_jsonl(fintabnet_rootpath, eachjsonlfile)

    return


if __name__ == "__main__":
    main(args.source_path)
