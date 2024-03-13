"""
CARTE
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
"""
import os
import xml.dom.minidom

from data_structure import AllTables, Table


def eval_carte(arg):
    gt_file, pred_path, match_text, show_flag = arg
    pred_file = os.path.join(pred_path, os.path.basename(gt_file))

    if not os.path.exists(pred_file):
        print(f"prediction file {pred_file} not found...")
        raise ValueError

    carte = Carte(gt_file, pred_file, match_text)
    single_stat = carte.eval()
    single_stat._filename = gt_file

    if show_flag:
        print(gt_file)
        single_stat.print_detail()

    return single_stat


class Statistics:
    def __init__(self, dict=None):
        self._filename = ""

        self._gt = 0
        self._pred = 0
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._dc = 0

        self._gt_ex = 0
        self._gt_im = 0
        self._tp_ex = 0
        self._tp_im = 0

        if dict is not None:
            self._gt = dict["gt"]
            self._pred = dict["pred"]
            self._tp = dict["tp"]
            self._fp = dict["fp"]
            self._fn = dict["fn"]
            self._dc = dict["dc"]

            self._gt_ex = dict["gt_ex"]
            self._gt_im = dict["gt_im"]
            self._tp_ex = dict["tp_ex"]
            self._tp_im = dict["tp_im"]


    def update(self, stat):
        self._gt += stat._gt
        self._pred += stat._pred
        self._tp += stat._tp
        self._fp += stat._fp
        self._fn += stat._fn
        self._dc += stat._dc

        self._gt_ex += stat._gt_ex
        self._gt_im += stat._gt_im
        self._tp_ex += stat._tp_ex
        self._tp_im += stat._tp_im


    def calc(self):
        recall = float(self._tp) / self._gt if self._gt else 0
        precision = float(self._tp) / self._pred if self._pred else 0
        hmean = (
            2 * recall * precision / (recall + precision) if (recall + precision) else 0
        )

        return {"recall": recall, "precision": precision, "f1": hmean}

    def calc_line(self):
        recall_ex = float(self._tp_ex) / self._gt_ex if self._gt_ex else 0
        recall_im = float(self._tp_im) / self._gt_im if self._gt_im else 0

        return {"recall_ex": recall_ex, "recall_im": recall_im}

    def print(self):
        ret = self.calc()
        print(
            f"recall:{ret['recall']:.4f}, precision:{ret['precision']:.4f}, f1:{ret['f1']:.4f}"
        )

    def print_detail(self):
        ret = self.calc()
        ret_line = self.calc_line()
        print(
            f"gt:{self._gt}, pred:{self._pred}, tp:{self._tp}, fp:{self._fp}, fn:{self._fn}, dc:{self._dc} || gt_ex:{self._gt_ex}, tp_ex:{self._tp_ex}, gt_im:{self._gt_im}, tp_im:{self._tp_im} || recall_ex:{ret_line['recall_ex']:.4f}, recall_im:{ret_line['recall_im']:.4f}"
        )
        print(
            f"recall:{ret['recall']:.4f}, precision:{ret['precision']:.4f}, f1:{ret['f1']:.4f}"
        )


class Carte:
    def __init__(self, gt_path, res_path, match_text):
        self.GTFile = gt_path
        self.resultFile = res_path
        self.match_text = match_text

    def eval(self):
        gt_dom = xml.dom.minidom.parse(self.GTFile)

        # incorrect submission format handling
        try:
            pred_dom = xml.dom.minidom.parse(self.resultFile)
        except Exception as e:
            print(f"check result format{e}")
            raise ValueError

        ret = self.evaluate_table(gt_dom, pred_dom, self.match_text)
        return ret

    @staticmethod
    def get_table_list(dom):
        """
        return a list of Table objects corresponding to the table element of the DOM.
        """
        return [Table(_nd) for _nd in dom.documentElement.getElementsByTagName("table")]

    @staticmethod
    def evaluate_table(gt_dom, pred_dom, match_text=False):
        # parse the tables in input elements
        gt_table_list = Carte.get_table_list(gt_dom)
        pred_table_list = Carte.get_table_list(pred_dom)

        # Combine all tables
        gt_tables = AllTables(gt_table_list)
        pred_tables = AllTables(pred_table_list)

        # Calculate scores
        dict_stat = gt_tables.calculate_score(pred_tables, match_text)

        return Statistics(dict_stat)

    def parse_gt(self):
        gt_dom = xml.dom.minidom.parse(self.GTFile)
        gt_table_list = Carte.get_table_list(gt_dom)
        gt_tables = AllTables(gt_table_list)

        return gt_tables


