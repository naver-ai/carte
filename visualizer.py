"""
CARTE
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
"""
import argparse
import glob
import os

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from carte import Carte

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


parser = argparse.ArgumentParser(description="CARTE table visualizer")
parser.add_argument("-g", "--gt_path", default="./sample_gt/", help="Path of the ground truth files.")
parser.add_argument("-i", "--img_path", default=None, help="Path of the image files. (PDF supported) If not specified, it will be the same as gt_path.")
parser.add_argument("--dpi", default=150, help="DPI of the PDF file for conversion (SciTSR:150, FinTabNet:72)")
parser.add_argument("--save_path", default="./results_vis/")
args = parser.parse_args()

# find gt files
if os.path.isdir(args.gt_path):
    gt_file_lst = glob.glob(args.gt_path + "/*.xml")
else:
    gt_file_lst = [args.gt_path]

# find image or pdf files
args.img_path = args.gt_path if not args.img_path else args.img_path
if os.path.isdir(args.img_path):
    input_type = 'img'
    image_list = get_image_list(args.img_path)
if len(image_list) == 0:
    input_type = 'pdf'
    image_list = get_pdf_list(args.img_path)

# prepare save path
os.makedirs(args.save_path, exist_ok=True)

# visualize table annotations
for gt_file in tqdm(gt_file_lst, total=len(gt_file_lst)):
    carte = Carte(gt_file, None, None)
    gt_tables = carte.parse_gt()

    img_basename = os.path.splitext(os.path.basename(gt_file))[0]
    if '__' in img_basename:    # for FinTabNet dataset
        img_basename_compare = img_basename.replace('__', '/')
    else:
        img_basename_compare = img_basename

    img_filename = None
    for img_file in image_list:
        if img_basename_compare+"." in img_file:
            img_filename = img_file
            break

    if img_filename:
        if input_type == 'img':
            img = cv2.imread(img_filename, cv2.IMREAD_UNCHANGED)        # BGR order
        elif input_type == 'pdf':
            pdf_pages = convert_from_path(img_filename, args.dpi)
            img = np.asarray(pdf_pages[0], dtype='int32').astype(np.uint8)    # 1 page
    else:
        print(f'No file found({img_file})...')
        continue

    renderTables(img_basename, img, gt_tables, dirname=args.save_path)

