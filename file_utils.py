"""
CARTE
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
"""
# -*- coding: utf-8 -*-
import glob
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTENTIONS = ['jpg', 'jpeg', 'png', 'JPG', 'tiff', 'TIFF']


# file listing functions
def get_image_list(dir):
    image_list = []
    for ext in IMAGE_EXTENTIONS:
        image_list.extend(get_file_list(dir, ext))
    return image_list

def get_pdf_list(dir):
    image_list = []
    for ext in ['pdf']:
        image_list.extend(get_file_list(dir, ext))
    return image_list

def get_xml_list(dir):
    image_list = []
    for ext in ['xml']:
        image_list.extend(get_file_list(dir, ext))
    return image_list

def get_file_list(dir, ext):
    return glob.glob("%s/**/*.%s" % (dir, ext), recursive=True)


# rendering functions
def renderTables(filename, img, tables, page=None, dirname='./results/'):
    img = np.array(img).astype(np.float32)

    # make result file list
    page = '' if page is None else '-' + page
    vis_filename = os.path.join(dirname, filename + page + '.jpg')

    fill_img = img.copy()
    for cell in tables._celladj:
        poly = cell._cell_box.reshape(-1, 2)
        # t, b, l, r
        line_type = cell._line_type
        line_color = [(255, 0, 0), (0, 0, 255)]        
        cv2.line(img, poly[0], poly[1], color=line_color[line_type[0]], thickness=line_type[0]+1)
        cv2.line(img, poly[2], poly[3], color=line_color[line_type[1]], thickness=line_type[1]+1)
        cv2.line(img, poly[0], poly[3], color=line_color[line_type[2]], thickness=line_type[2]+1)
        cv2.line(img, poly[1], poly[2], color=line_color[line_type[3]], thickness=line_type[3]+1)
        cv2.fillPoly(fill_img, [poly], (0, 128, 0), 8)

    for cell in tables._empty_celladj:
        poly = cell._cell_box.reshape(-1, 2)
        cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(255, 0, 255), thickness=1)

    # Optionally render row/col IDs
    try:
        fontpath = "./Arial-Unicode-Bold.ttf"
        font = ImageFont.truetype(fontpath, 15)

        pil_img = Image.fromarray(img.astype(np.uint8))
        draw = ImageDraw.Draw(pil_img)
        for cell in tables._celladj:
            pos = cell._cell_box.reshape(-1, 2)[0]
            # text = f"r{str(cell._rows)} c{str(cell._cols)}"
            text = f"r{cell._rows[0]} c{cell._cols[0]}"
            draw.text(pos, text, font=font, fill=(0, 128, 0))
        for cell in tables._empty_celladj:
            pos = cell._cell_box.reshape(-1, 2)[0]
            # text = f"r{str(cell._rows)} c{str(cell._cols)}"
            text = f"r{cell._rows[0]} c{cell._cols[0]}"
            draw.text(pos, text, font=font, fill=(0, 128, 0))
        img = np.array(pil_img).astype(np.float32)
    except Exception as e:
        pass

    cv2.addWeighted(fill_img, 0.2, img, 0.8, 0, img)

    # Save result image
    cv2.imwrite(vis_filename, img)

