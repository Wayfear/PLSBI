# Partial Least Square Regression for PTSD


The repo is the open-source code for the [paper]().

## Dataset

### Brain Imaging

The path of the imaging file is provided by the parameter "--imaging". The brain imaging file should be an RData file containing two variables, "FC" and "subjid". "FC" is a group of functional connectivities stored as a 3D matrix. The last dimension of the 3D matrix is the sample size. For example, in our dataset, the size of "FC" is (279, 279, 98). "subjid" is a list containing all subject's id in "FC". The sample size of the 3D matrix should be equal to the length of "subjid" and each id in the list corresponds with a functional connectivity in order.

### Clinical Labels

The path of the label file that should contain a column named "subjid" is provided by the parameter "--clinical_file". These columns used as the prediction labels are provided by the parameter "--columns". For example, "--columns ptsdss1_categorical ptsdss2_categorical ptsdss3_categorical" can be used for our dataset.

The label file should be a CSV file, splited by ",". After parsing the label file. These columns specified by the parameter "--columns" in the label file will be used to fit the PLS model.


## Usage

```
usage: main.py [-h] [--output OUTPUT] [--imaging IMAGING]
               [--clinical_file CLINICAL_FILE] [--column COLUMN]
               [--correlation_threshold CORRELATION_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT       The folder for storing results
  --imaging IMAGING     The file contains imaging data, the format is RData
  --clinical_file CLINICAL_FILE
                        The file contains clinical variables, the format is
                        csv
  --columns COLUMNS     These columns used as the prediction labels
  --correlation_threshold CORRELATION_THRESHOLD
                        The threshold used to select correlated edges
```

## Output

### The process of finding the optimal component number
![](result/suggest_com_num.png)

### Performance

R^2: 0.6358

### Y loading

|     column      | comp 1 | comp 2 | comp 3 | comp 4 | comp 5 |
|:---------------:|:------:|:------:|:------:|:------:|:------:|
|    INTRUSIVE    | 0.090  | 0.064  | -0.033 | 0.109  | -0.048 |
|    AVOIDANCE    | 0.082  | 0.100  | 0.087  | -0.004 | -0.065 |
| NEGATIVE AFFECT | 0.100  | 0.056  | 0.023  | 0.039  | 0.090  |
|   HYERAROUSAL   | 0.086  | 0.076  | -0.106 | -0.039 | 0.008  |


### Identify significant edges by X loading

Significant edges for each component can be found in files whose paths are "result/original_{component index}_by_rank.edge".