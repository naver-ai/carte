# CARTE: Cell Adjacency Relation for Table Evaluation

Welcome to the official repository of CARTE, a comprehensive evaluation metric for table recognition. CARTE offers a holistic approach to assessing table recognition performance including table detection (TD) and Table Structure Recognition(TSR). Building upon the evaluation methodology used in the ICDAR19 Table competition, CARTE introduces (1) a parameter-free cell matching approach and (2) a simplified scoring mechanism based on counting the adjacency relations of matched cells.

## Updates
**_2024-03-13_**: Initial commit.

### Requirements
* Python >= 3.7
* Check the requirements.txt file for package dependencies. To install them, use the following command:
```
pip install -r requirements.txt
```

## Instructions for Standalone Scripts
### Table Evaluation
To evaluate the entire sample in the dataset:
```
python eval.py --gt_path (GT_PATH) --pred_path (PRED_PATH)
```

To evaluate a single file:
```
python eval.py --gt_path (GT_PATH)/(GT_FILE) --pred_path (PRED_PATH)
```
In this case, the evaluator will find the corresponding file in (PRED_PATH).

### Toy Example
Try this command:
```
python eval.py --gt_path sample_gt --pred_path sample_result
```

### Table Data Conversion
To evaluate datasets using the CARTE metric, you need to convert their data format. We provide converters for published datasets to make them compatible with CARTE. We support SciTSR, WTW, and FinTabNet for data conversion. Additionally, you can evaluate the IC19 dataset without data conversion. Below are the instructions for converting the test set for evaluation, but you can adapt these codes for the training set as well.

For [SciTSR](https://github.com/Academic-Hammer/SciTSR):
```
python converter/scitsr2xml.py -s /data/db/table/scitsr/test
```

For [FinTabNet](https://developer.ibm.com/exchanges/data/all/fintabnet/):
```
python converter/fintab2xml.py -s /data/db/table/fintabnet
```

For [WTW](https://github.com/wangwen-whu/WTW-Dataset):
```
python converter/wtw2xml.py -s /data/db/table/WTW -t /data/db/table/WTWxml
```
Please use the revised version of XML for WTW. Refer to the official website for details.

### Visualization
You can visualize table datasets using the following commands:

For SubTableBank:
```
python visualizer.py -g /data/db/table/SubTableBank/test/
```

For SciTSR:
```
python visualizer.py -g /data/db/table/scitsr/test/xml -i /data/db/table/scitsr/test/pdf
```

For FinTabNet:
```
python visualizer.py -g /data/db/table/fintabnet/xml/table/test/ -i /data/db/table/fintabnet/pdf/ --dpi 72
```

For WTW:
```
python visualizer.py -g /data/db/table/WTWCell/test/
```

## Data Format
The result files should adhere to the same XML format as the ICDAR 2019 table competition. Refer to the {sample_gt} and {sample_result} directories.

## Acknowledgment

We extend our sincere gratitude to [IC19](https://github.com/cndplab-founder/ctdar_measurement_tool) for providing evaluation metrics.

## Citation
```
(TBU)
```

## License
```
Copyright (c) 2024-present NAVER Cloud Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
