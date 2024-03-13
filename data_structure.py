"""
Data structures used by the evaluation process.

Yu Fang - March 2019
"""

import os
import xml.dom.minidom
from collections import Iterable
from difflib import SequenceMatcher

import numpy as np
from shapely.geometry import MultiPoint, Polygon


# helper functions
def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


# derived from https://blog.csdn.net/u012433049/article/details/82909484
def compute_poly_iou(list1, list2):
    a1 = np.array(list1, dtype=int).reshape(-1, 2)
    poly1 = Polygon(a1)
    poly1_clean = poly1.buffer(0)

    a2 = np.array(list2, dtype=int).reshape(-1, 2)
    poly2 = Polygon(a2)
    poly2_clean = poly2.buffer(0)

    try:
        # iou = poly1.intersection(poly2).area / poly1.union(poly2).area
        iou = (
            poly1_clean.intersection(poly2_clean).area
            / poly1_clean.union(poly2_clean).area
        )
    except ZeroDivisionError:
        iou = 0
    return iou


def cellbox_to_array(box):
    temp = []
    for el in box.split():
        temp.append((el.split(",")))
    cell_box = list(flatten(temp))
    cell_box = [int(float(x)) for x in cell_box]

    return np.array(cell_box, dtype=int).reshape(-1, 2)


def compute_poly_overlap(a1, a2):
    poly1 = Polygon(a1)
    poly1_clean = poly1.buffer(0)
    poly2 = Polygon(a2)
    poly2_clean = poly2.buffer(0)

    try:
        overlap = poly1_clean.intersection(poly2_clean).area / poly1_clean.area
    except ZeroDivisionError:
        overlap = 0
    return overlap


class CellAdj:
    def __init__(self, cell):
        self._cell_box = cell._cell_box
        self._minx = min(self._cell_box[:, 0])
        self._miny = min(self._cell_box[:, 1])
        self._maxx = max(self._cell_box[:, 0])
        self._maxy = max(self._cell_box[:, 1])
        self._table_id = cell._table_id  # the table_id this cell belongs to
        self._cell_id = cell._cell_id
        self._content_text = cell._content_text
        self._rows = (cell._start_row, cell._end_row)
        self._cols = (cell._start_col, cell._end_col)
        self._line_type = cell._line_type

        self._num_hor_adj = 0
        self._num_ver_adj = 0

    def __str__(self):
        return "CELL Adjency object - {} hor, {} ver".format(
            self._num_hor_adj, self._num_ver_adj
        )


class Cell(object):
    # @:param start_row : start row index of the Cell
    # @:param start_col : start column index of the Cell
    # @:param end-row : end row index of the Cell
    # @:param end-col : end column index of the Cell
    # @:param cell_box: bounding-box of the Cell (coordinates are saved as a string)
    # @:param content_text: the text content within Cell
    # @:param cell_id: unique id of the Cell

    def __init__(
        self,
        table_id,
        start_row,
        start_col,
        cell_box_str,
        cell_box,
        end_row,
        end_col,
        content_text="",
        line_type=[1, 1, 1, 1]
    ):
        self._start_row = int(start_row)
        self._start_col = int(start_col)
        self._cell_box_str = cell_box_str
        self._cell_box = cell_box
        self._content_text = content_text
        self._table_id = table_id  # the table_id this cell belongs to
        # self._cell_name = cell_id    # specify the cell using passed-in cell_id
        self._cell_id = id(self)
        self._line_type = line_type     # TOP, BOTTOM, LEFT, RIGHT
        # self._region = region

        # check for end-row and end-col special case
        if end_row == -1:
            self._end_row = self.start_row
        else:
            self._end_row = int(end_row)
        if end_col == -1:
            self._end_col = self._start_col
        else:
            self._end_col = int(end_col)

    @property
    def start_row(self):
        return self._start_row

    @property
    def start_col(self):
        return self._start_col

    @property
    def end_row(self):
        return self._end_row

    @property
    def end_col(self):
        return self._end_col

    @property
    def cell_box(self):
        return self._cell_box

    @property
    def cell_id(self):
        return self._cell_id

    @property
    def table_id(self):
        return self._table_id

    def __str__(self):
        return "CELL row=[%d, %d] col=[%d, %d] (coords=%s)" % (
            self.start_row,
            self.end_row,
            self.start_col,
            self.end_col,
            self.cell_box,
        )

    # return the IoU value of two cell blocks
    def compute_cell_iou(self, another_cell):
        cell_box_1_temp = []
        for el in self.cell_box.split():
            cell_box_1_temp.append((el.split(",")))
        cell_box_1 = list(flatten(cell_box_1_temp))
        cell_box_1 = [int(x) for x in cell_box_1]

        cell_box_2_temp = []
        for el in another_cell.cell_box.split():
            cell_box_2_temp.append((el.split(",")))
        cell_box_2 = list(flatten(cell_box_2_temp))
        cell_box_2 = [int(x) for x in cell_box_2]

        return compute_poly_iou(cell_box_1, cell_box_2)

    # check if the two cell object denotes same cell area in table
    def check_same(self, another_cell):
        return (
            self._start_row == another_cell.start_row
            and self._end_row == another_cell.end_row
            and self._start_col == another_cell.start_col
            and self._end_col == another_cell.end_col
        )


# Note: currently save the relation with two cell object involved,
# can be replaced by cell_id in follow-up memory clean up
class AdjRelation:
    DIR_HORIZ = 1
    DIR_VERT = 2
    DIR_NONE = 3

    def __init__(self, fromText, toText, direction):
        # @param: fromText, toText are Cell objects （may be changed to cell-ID for further development）
        self._fromText = fromText
        self._toText = toText
        self._direction = direction

    @property
    def fromText(self):
        return self._fromText

    @property
    def toText(self):
        return self._toText

    @property
    def direction(self):
        return self._direction

    def __str__(self):
        if self.direction == self.DIR_VERT:
            dir = "vertical"
        elif self.direction == self.DIR_HORIZ:
            dir = "horizontal"
        else:
            dir = "none"
        return (
            "ADJ_RELATION: "
            + str(self._fromText)
            + "  "
            + str(self._toText)
            + "    "
            + dir
        )

    def isEqual(self, otherRelation):
        return (
            self.fromText.cell_id == otherRelation.fromText.cell_id
            and self.toText.cell_id == otherRelation.toText.cell_id
            and self.direction == otherRelation.direction
        )


class Table:
    def __init__(self, tableNode):
        self._root = tableNode
        self._id = id(self)
        self._table_coords = ""
        self._maxRow = 0  # PS: indexing from 0
        self._maxCol = 0
        self._cells = []  # save a table as list of <Cell>s
        self._empty_cells = []  # save a table as list of <Cell>s
        self.adj_relations = []  # save the adj_relations for the table
        self.parsed = False
        self.found = False  # check if the find_adj_relations() has been called once

        self.parse_table()  # including finding empty cells

    def __str__(self):
        return "TABLE object - {} row x {} col".format(
            self._maxRow + 1, self._maxCol + 1
        )

    @property
    def id(self):
        return self._id

    @property
    def table_coords(self):
        return self._table_coords

    @property
    def table_cells(self):
        return self._cells

    # parse input xml to cell lists
    def parse_table(self):
        MAX_CELL_NUM = 1000
        cell_flag = [[False for _ in range(MAX_CELL_NUM)] for _ in range(MAX_CELL_NUM)]
        row_min = [9999] * MAX_CELL_NUM
        row_max = [-9999] * MAX_CELL_NUM
        col_min = [9999] * MAX_CELL_NUM
        col_max = [-9999] * MAX_CELL_NUM

        # get the table bbox
        self._table_coords = str(
            self._root.getElementsByTagName("Coords")[0].getAttribute("points")
        )

        # get info for each cell
        cells = self._root.getElementsByTagName("cell")
        max_row = max_col = 0
        min_row = min_col = 9999
        for cell in cells:
            sr = cell.getAttribute("start-row")
            sc = cell.getAttribute("start-col")
            cell_id = cell.getAttribute("id")
            b_points = str(
                cell.getElementsByTagName("Coords")[0].getAttribute("points")
            )
            # try:
            #     try:
            #         text = cell.getElementsByTagName("content")[0].firstChild.nodeValue
            #     except AttributeError:
            #         text = ""
            # except IndexError:
            #     text = "initialized cell as no content"
            er = cell.getAttribute("end-row") if cell.hasAttribute("end-row") else -1
            ec = cell.getAttribute("end-col") if cell.hasAttribute("end-col") else -1
            if cell.getElementsByTagName("Content").length == 0:
                ct = ""
            else:
                ct = str(cell.getElementsByTagName("Content")[0].getAttribute("text"))
            if cell.getElementsByTagName("Lines").length == 0:
                line_type = [1, 1, 1, 1]
            else:   # top, bottom, left, right order
                line_type = [int(cell.getElementsByTagName("Lines")[0].getAttribute("top")),
                             int(cell.getElementsByTagName("Lines")[0].getAttribute("bottom")),
                             int(cell.getElementsByTagName("Lines")[0].getAttribute("left")),
                             int(cell.getElementsByTagName("Lines")[0].getAttribute("right"))]
            
            cell_box = cellbox_to_array(b_points)
            new_cell = Cell(
                table_id=str(self.id),
                start_row=sr,
                start_col=sc,
                cell_box_str=b_points,
                cell_box = cell_box,
                end_row=er,
                end_col=ec,
                content_text=ct,
                line_type=line_type
            )

            sc, ec, sr, er = int(sc), int(ec), int(sr), int(er)
            col_min[sc] = min(col_min[sc], cell_box[:,0].min())
            col_max[ec] = max(col_max[ec], cell_box[:,0].max())
            row_min[sr] = min(row_min[sr], cell_box[:,1].min())
            row_max[er] = max(row_max[er], cell_box[:,1].max())

            for r in range(sr, er + 1):
                for c in range(sc, ec + 1):
                    cell_flag[r][c] = True

            max_row = max(max_row, sr, er)
            max_col = max(max_col, sc, ec)
            min_row = min(min_row, sr, er)
            min_col = min(min_col, sc, ec)
            self._cells.append(new_cell)
        self._maxCol = max_col
        self._maxRow = max_row

        # calc cell boundaries for empty cells
        while(True):    # row-wise traverse
            changed = False
            for r in range(min_row, max_row + 1):
                if r != min_row:
                    if abs(row_min[r]) == 9999:
                        row_min[r] = row_max[r-1]
                        if abs(row_min[r]) != 9999:
                            changed = True
                if r != max_row:
                    if abs(row_max[r]) == 9999:
                        row_max[r] = row_min[r+1]
                        if abs(row_max[r]) != 9999:
                            changed = True
            if not changed:
                break
        while(True):    # col-wise traverse
            changed = False
            for c in range(min_col, max_col + 1):
                if c != min_col:
                    if abs(col_min[c]) == 9999:
                        col_min[c] = col_max[c-1]
                        if abs(col_min[c]) != 9999:
                            changed = True
                if c != max_col:
                    if abs(col_max[c]) == 9999:
                        col_max[c] = col_min[c+1]
                        if abs(col_max[c]) != 9999:
                            changed = True
            if not changed:
                break

        # find empty cells
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                if cell_flag[r][c]: continue

                b_points = "(EmptyCell)"
                x1, y1, x2, y2 = col_min[c], row_min[r], col_max[c], row_max[r]

                if 9999 in [abs(x1), abs(x2), abs(y1), abs(y2)]:
                    print(f'cannot make empty cell... (r={r}, c={c}), ({x1},{y1},{x2},{y2})')
                    continue

                cell_box = np.array([x1, y1, x2, y1, x2, y2, x1, y2]).reshape(-1, 2)
                self._empty_cells.append(
                    Cell(
                        table_id=str(self.id),
                        start_row=str(r),
                        start_col=str(c),
                        cell_box_str=b_points,
                        cell_box=cell_box,
                        end_row=str(r),
                        end_col=str(c),
                        content_text="",
                ))

        self.parsed = True

    # generate a table-like structure for finding adj_relations
    def convert_2d(self):
        table = [
            [0 for x in range(self._maxCol + 1)] for y in range(self._maxRow + 1)
        ]  # init blank cell with int 0
        for cell in self._cells:
            cur_row = cell.start_row
            while cur_row <= cell.end_row:
                cur_col = cell.start_col
                while cur_col <= cell.end_col:
                    temp = table[cur_row][cur_col]
                    if temp == 0:
                        table[cur_row][cur_col] = cell
                    elif type(temp) == list:
                        temp.append(cell)
                        table[cur_row][cur_col] = temp
                    else:
                        table[cur_row][cur_col] = [temp, cell]
                    cur_col += 1
                cur_row += 1

        return table

    def find_adj_relations(self):
        if self.found:
            return self.adj_relations
        else:
            # if len(self._cells) == 0:
            if self.parsed == False:
                # fix: cases where there's no cell in table?
                print("table is not parsed for further steps.")
                self.parse_table()
                self.find_adj_relations()
            else:
                retVal = []
                tab = self.convert_2d()

                # find horizontal relations
                for r in range(self._maxRow + 1):
                    for c_from in range(self._maxCol):
                        temp_pos = tab[r][c_from]
                        if temp_pos == 0:
                            continue
                        elif type(temp_pos) == list:
                            for cell in temp_pos:
                                c_to = c_from + 1
                                if tab[r][c_to] != 0:
                                    # find relation between two adjacent cells
                                    if type(tab[r][c_to]) == list:
                                        for cell_to in tab[r][c_to]:
                                            if cell != cell_to and (
                                                not cell.check_same(cell_to)
                                            ):
                                                adj_relation = AdjRelation(
                                                    cell, cell_to, AdjRelation.DIR_HORIZ
                                                )
                                                retVal.append(adj_relation)
                                    else:
                                        if cell != tab[r][c_to]:
                                            adj_relation = AdjRelation(
                                                cell,
                                                tab[r][c_to],
                                                AdjRelation.DIR_HORIZ,
                                            )
                                            retVal.append(adj_relation)
                                else:
                                    # find the next non-blank cell, if exists
                                    for temp in range(c_from + 1, self._maxCol + 1):
                                        if tab[r][temp] != 0:
                                            if type(tab[r][temp]) == list:
                                                for cell_to in tab[r][temp]:
                                                    adj_relation = AdjRelation(
                                                        cell,
                                                        cell_to,
                                                        AdjRelation.DIR_HORIZ,
                                                    )
                                                    retVal.append(adj_relation)
                                            else:
                                                adj_relation = AdjRelation(
                                                    cell,
                                                    tab[r][temp],
                                                    AdjRelation.DIR_HORIZ,
                                                )
                                                retVal.append(adj_relation)
                                            break
                        else:
                            c_to = c_from + 1
                            if tab[r][c_to] != 0:
                                # find relation between two adjacent cells
                                if type(tab[r][c_to]) == list:
                                    for cell_to in tab[r][c_to]:
                                        if temp_pos != cell_to:
                                            adj_relation = AdjRelation(
                                                temp_pos, cell_to, AdjRelation.DIR_HORIZ
                                            )
                                            retVal.append(adj_relation)
                                else:
                                    if temp_pos != tab[r][c_to]:
                                        adj_relation = AdjRelation(
                                            temp_pos,
                                            tab[r][c_to],
                                            AdjRelation.DIR_HORIZ,
                                        )
                                        retVal.append(adj_relation)
                            else:
                                # find the next non-blank cell, if exists
                                for temp in range(c_from + 1, self._maxCol + 1):
                                    if tab[r][temp] != 0:
                                        if type(tab[r][temp]) == list:
                                            for cell_to in tab[r][temp]:
                                                adj_relation = AdjRelation(
                                                    temp_pos,
                                                    cell_to,
                                                    AdjRelation.DIR_HORIZ,
                                                )
                                                retVal.append(adj_relation)
                                        else:
                                            adj_relation = AdjRelation(
                                                temp_pos,
                                                tab[r][temp],
                                                AdjRelation.DIR_HORIZ,
                                            )
                                            retVal.append(adj_relation)
                                        break

                # find vertical relations
                for c in range(self._maxCol + 1):
                    for r_from in range(self._maxRow):
                        temp_pos = tab[r_from][c]
                        if temp_pos == 0:
                            continue
                        elif type(temp_pos) == list:
                            for cell in temp_pos:
                                r_to = r_from + 1
                                if tab[r_to][c] != 0:
                                    # find relation between two adjacent cells
                                    if type(tab[r_to][c]) == list:
                                        for cell_to in tab[r_to][c]:
                                            if cell != cell_to and (
                                                not cell.check_same(cell_to)
                                            ):
                                                adj_relation = AdjRelation(
                                                    cell, cell_to, AdjRelation.DIR_VERT
                                                )
                                                retVal.append(adj_relation)
                                    else:
                                        if cell != tab[r_to][c]:
                                            adj_relation = AdjRelation(
                                                cell, tab[r_to][c], AdjRelation.DIR_VERT
                                            )
                                            retVal.append(adj_relation)
                                else:
                                    # find the next non-blank cell, if exists
                                    for temp in range(r_from + 1, self._maxRow + 1):
                                        if tab[temp][c] != 0:
                                            if type(tab[temp][c]) == list:
                                                for cell_to in tab[temp][c]:
                                                    adj_relation = AdjRelation(
                                                        cell,
                                                        cell_to,
                                                        AdjRelation.DIR_VERT,
                                                    )
                                                    retVal.append(adj_relation)
                                            else:
                                                adj_relation = AdjRelation(
                                                    cell,
                                                    tab[temp][c],
                                                    AdjRelation.DIR_VERT,
                                                )
                                                retVal.append(adj_relation)
                                            break
                        else:
                            r_to = r_from + 1
                            if tab[r_to][c] != 0:
                                # find relation between two adjacent cells
                                if type(tab[r_to][c]) == list:
                                    for cell_to in tab[r_to][c]:
                                        if temp_pos != cell_to:
                                            adj_relation = AdjRelation(
                                                temp_pos, cell_to, AdjRelation.DIR_VERT
                                            )
                                            retVal.append(adj_relation)
                                else:
                                    if temp_pos != tab[r_to][c]:
                                        adj_relation = AdjRelation(
                                            temp_pos, tab[r_to][c], AdjRelation.DIR_VERT
                                        )
                                        retVal.append(adj_relation)
                            else:
                                # find the next non-blank cell, if exists
                                for temp in range(r_from + 1, self._maxRow + 1):
                                    if tab[temp][c] != 0:
                                        if type(tab[temp][c]) == list:
                                            for cell_to in tab[temp][c]:
                                                adj_relation = AdjRelation(
                                                    temp_pos,
                                                    cell_to,
                                                    AdjRelation.DIR_VERT,
                                                )
                                                retVal.append(adj_relation)
                                        else:
                                            adj_relation = AdjRelation(
                                                temp_pos,
                                                tab[temp][c],
                                                AdjRelation.DIR_VERT,
                                            )
                                            retVal.append(adj_relation)
                                        break

                # eliminate duplicates
                # repeat = True
                # while repeat:
                #     repeat = False
                #     duplicates = []

                #     for ar1 in retVal:
                #         for ar2 in retVal:
                #             if ar1 != ar2:
                #                 if (
                #                     ar1.direction == ar2.direction
                #                     and ar1.fromText == ar2.fromText
                #                     and ar1.toText == ar2.toText
                #                 ):
                #                     duplicates.append(ar2)
                #                     break
                #         else:
                #             continue
                #         break

                #     if len(duplicates) > 0:
                #         repeat = True
                #         retVal.remove(duplicates[0])
                duplicates_index = set()
                for m in range(len(retVal) - 1):
                    ar1 = retVal[m]
                    direction = ar1.direction
                    fromText = ar1.fromText
                    toText = ar1.toText
                    for n in range(m + 1, len(retVal)):
                        ar2 = retVal[n]
                        if (
                            direction == ar2.direction
                            and fromText == ar2.fromText
                            and toText == ar2.toText
                        ):
                            duplicates_index.add(n)  

                duplicates_index = list(duplicates_index)
                duplicates_index.sort(reverse=True)
                for k in duplicates_index:
                    # print(k, len(retVal))
                    del retVal[k]

                self.found = True
                self.adj_relations = retVal
            return self.adj_relations

    # compute the IOU of table, pass-in var is another Table object
    def compute_table_iou(self, another_table):
        table_box_1_temp = []
        for el in self.table_coords.split():
            table_box_1_temp.append((el.split(",")))
        table_box_1 = list(flatten(table_box_1_temp))
        table_box_1 = [int(x) for x in table_box_1]

        table_box_2_temp = []
        for el in another_table.table_coords.split():
            table_box_2_temp.append((el.split(",")))
        table_box_2 = list(flatten(table_box_2_temp))
        table_box_2 = [int(x) for x in table_box_2]

        return compute_poly_iou(table_box_1, table_box_2)

    # find the cell mapping of tables as dictionary, pass-in var is another table and the desired IOU value
    def find_cell_mapping(self, target_table, iou_value):
        mapped_cell = (
            []
        )  # store the matches as tuples - (gt, result) mind the order of table when passing in
        for cell_1 in self.table_cells:
            for cell_2 in target_table.table_cells:
                if cell_1.compute_cell_iou(cell_2) >= iou_value:
                    mapped_cell.append((cell_1, cell_2))
                    break
        ret = dict(mapped_cell)
        # print(ret)
        return ret

    # to print a table cell mapping
    @classmethod
    def printCellMapping(cls, dMappedCell):
        print("-" * 25)
        for cell1, cell2 in dMappedCell.items():
            print("  ", cell1, " --> ", cell2)

    # to print a table set of adjacency relations
    @classmethod
    def printAdjacencyRelationList(cls, lAdjRel, title=""):
        print("--- %s " % title + "-" * 25)
        for adj in lAdjRel:
            print(adj)


class AllTables:
    def __init__(self, tableList):
        self._celladj = list()
        self._empty_celladj = list()
        self._tableList = tableList

        # cell matching
        self._matched = False
        self._match_idx = list()

        self.parse_cell_adj()
        self.parse_empty_cell_adj()

    def parse_cell_adj(self):
        celladj_dict = dict()

        for table in self._tableList:
            for adj in table.find_adj_relations():
                # retrieve adjency data
                fromCell = adj._fromText

                # retrieve celladj
                cell_id = fromCell._cell_id
                celladj = celladj_dict.get(cell_id, None)
                if celladj is None:
                    celladj = CellAdj(fromCell)

                # increase adj num
                if adj._direction == 1:  # "horizontal":
                    celladj._num_hor_adj += 1
                elif adj._direction == 2:  # "vertical":
                    celladj._num_ver_adj += 1

                # replace
                celladj_dict[cell_id] = celladj

            # add cell without adj for matching candidate
            for cell in table._cells:
                if not cell._cell_id in celladj_dict:
                    celladj_dict[cell._cell_id] = CellAdj(cell)

        # dict() to list()
        for key, value in celladj_dict.items():
            self._celladj.append(value)

    def parse_empty_cell_adj(self):
        for table in self._tableList:
            for c in table._empty_cells:
                self._empty_celladj.append(CellAdj(
                    c                    
                ))

    def match_cells(self, A, B, match_text):
        if self._matched:
            return
        self._matched = True

        if not match_text:
            # find area recall & area precision
            matAR = self.calculate_overlap(A, B)
            matAP = self.calculate_overlap(B, A).transpose()
        else:
            # find text recall & text precision
            matAR = self.calculate_lcs(A, B)
            matAP = matAR.copy()
            # matAP = self.calculate_lcs(B, A).transpose()

        # matching matrix
        match_idx = [[] for _ in range(len(A))]
        matched_flag = [False] * len(B)

        if len(A) == 0 or len(B) == 0:
            self._match_idx = match_idx
            return

        # step1) match based on AR
        max_AR = np.argmax(matAR, axis=1)
        for m in range(len(A)):
            n = max_AR[m]
            if matAR[m, n] != 0:
                match_idx[m].append(n)
                matched_flag[n] = True

        # step2) match based on AP
        indSorted = np.dstack(
            np.unravel_index(np.argsort(-matAP, axis=None), matAP.shape)
        )[0]
        for m, n in indSorted:
            # no more overlap
            if matAP[m, n] == 0:
                break
            if matched_flag[n]:
                continue

            match_idx[m].append(n)
            matched_flag[n] = True

        self._match_idx = match_idx

    def match_empty_cells(self, B):
        A = self._empty_celladj
        # find area recall & area precision
        matAR = self.calculate_overlap(A, B)

        # matching matrix
        match_idx = [[] for _ in range(len(A))]

        if len(A) == 0 or len(B) == 0:
            self._match_empty_idx = match_idx
            return

        # step1) match based on AR
        max_AR = np.argmax(matAR, axis=1)
        for m in range(len(A)):
            n = max_AR[m]
            if matAR[m, n] != 0:
                match_idx[m].append(n)

        self._match_empty_idx = match_idx

    def calculate_overlap(self, A, B):
        len_A = len(A)
        len_B = len(B)
        mat_overlap = np.zeros((len_A, len_B), dtype=np.float32)

        for m in range(len_A):
            for n in range(len_B):
                a, b = A[m], B[n]
                # minimum requiremnt
                if (
                    a._minx > b._maxx
                    or a._miny > b._maxy
                    or a._maxx < b._minx
                    or a._maxy < b._miny
                ):
                    continue
                mat_overlap[m, n] = compute_poly_overlap(a._cell_box, b._cell_box)

        return mat_overlap

    def calculate_lcs(self, A, B):
        len_A = len(A)
        len_B = len(B)
        mat_overlap = np.zeros((len_A, len_B), dtype=np.float32)

        for m in range(len_A):
            for n in range(len_B):
                a, b = A[m], B[n]
                if len(a._content_text) == 0 and len(b._content_text) == 0:
                    mat_overlap[m, n] = 1
                    continue
                elif len(a._content_text) == 0:
                    continue
                s = SequenceMatcher(None, a._content_text, b._content_text)
                lcs = "".join(
                    [
                        a._content_text[block.a : (block.a + block.size)]
                        for block in s.get_matching_blocks()
                    ]
                )
                mat_overlap[m, n] = (
                    2 * len(lcs) / (len(a._content_text) + len(b._content_text))
                )

        return mat_overlap

    def merge_same_content(self, adjs):
        merged_adj = list()
        processed_texts = list()

        for adj in adjs:
            if adj._content_text in processed_texts:
                idx = processed_texts.index(adj._content_text)
                merged_adj[idx]._num_hor_adj += adj._num_hor_adj
                merged_adj[idx]._num_ver_adj += adj._num_ver_adj
            else:
                merged_adj.append(adj)
                processed_texts.append(adj._content_text)

        return merged_adj

    def calculate_score(self, res_tables, match_text):
        # merge cells with the same content
        if match_text:
            GTAdj = self.merge_same_content(self._celladj)
            PredAdj = self.merge_same_content(res_tables._celladj)
        else:
            GTAdj, PredAdj = self._celladj, res_tables._celladj

        # Match cells
        self.match_cells(GTAdj, PredAdj, match_text)
        self.match_empty_cells(PredAdj)

        # 1) Total GT, Pred
        tot_gt, tot_pred = 0, 0
        tot_gt_ex, tot_gt_im = 0, 0
        for adj in GTAdj:
            tot_gt += adj._num_hor_adj + adj._num_ver_adj
            if adj._line_type[1]:   # TBLR
                tot_gt_ex += adj._num_ver_adj
            else:
                tot_gt_im += adj._num_ver_adj
            if adj._line_type[3]:   # TBLR
                tot_gt_ex += adj._num_hor_adj
            else:
                tot_gt_im += adj._num_hor_adj
        for adj in PredAdj:
            tot_pred += adj._num_hor_adj + adj._num_ver_adj

        # 2) Match GT
        tot_tp, tot_dc = 0, 0
        tot_tp_ex, tot_tp_im = 0, 0
        cor_hor_adj, cor_ver_adj = 0, 0
        cor_hor_adj_ex, cor_ver_adj_ex = 0, 0
        match_flag = [False] * len(PredAdj)
        for m in range(len(GTAdj)):
            num_gt_hor_adj = GTAdj[m]._num_hor_adj
            num_gt_ver_adj = GTAdj[m]._num_ver_adj
            flag_ver_ex = GTAdj[m]._line_type[1]
            flag_hor_ex = GTAdj[m]._line_type[3]

            for n in self._match_idx[m]:
                match_flag[n] = True
                cur_cor_hor_adj = min(PredAdj[n]._num_hor_adj, num_gt_hor_adj)
                num_gt_hor_adj -= cur_cor_hor_adj
                PredAdj[n]._num_hor_adj -= cur_cor_hor_adj
                cor_hor_adj += cur_cor_hor_adj
                if flag_hor_ex: cor_hor_adj_ex += cur_cor_hor_adj

                cur_cor_ver_adj = min(PredAdj[n]._num_ver_adj, num_gt_ver_adj)
                num_gt_ver_adj -= cur_cor_ver_adj
                PredAdj[n]._num_ver_adj -= cur_cor_ver_adj
                cor_ver_adj += cur_cor_ver_adj
                if flag_ver_ex: cor_ver_adj_ex += cur_cor_ver_adj

        # 3) Match DC GT
        for m in range(len(self._empty_celladj)):
            for n in self._match_empty_idx[m]:
                if not match_flag[n]:
                    num_adj = PredAdj[n]._num_hor_adj + PredAdj[n]._num_ver_adj
                    tot_dc += num_adj
                    tot_pred -= num_adj

        tot_tp = cor_hor_adj + cor_ver_adj
        tot_tp_ex = cor_hor_adj_ex + cor_ver_adj_ex
        tot_tp_im = tot_tp - tot_tp_ex
        tot_fp = tot_pred - tot_tp
        tot_fn = tot_gt - tot_tp


        return {
            "gt": tot_gt,
            "pred": tot_pred,
            "tp": tot_tp,
            "fp": tot_fp,
            "fn": tot_fn,
            "dc": tot_dc,
            # additional stats
            "gt_ex": tot_gt_ex,
            "gt_im": tot_gt_im,
            "tp_ex": tot_tp_ex,
            "tp_im": tot_tp_im,
        }


class ResultStructure:
    def __init__(self, truePos, gtTotal, resTotal):
        self._truePos = truePos
        self._gtTotal = gtTotal
        self._resTotal = resTotal

    @property
    def truePos(self):
        return self._truePos

    @property
    def gtTotal(self):
        return self._gtTotal

    @property
    def resTotal(self):
        return self._resTotal

    def __str__(self):
        return "true: {}, gt: {}, res: {}".format(
            self._truePos, self._gtTotal, self._resTotal
        )
