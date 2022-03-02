# Partial Least Square Regression for PTSD


The repo is the open-source code for the [paper]().

## Usage

```
usage: main.py [-h] [--output OUTPUT] [--imaging IMAGING]
               [--clinical_file CLINICAL_FILE] [--id_file ID_FILE]
               [--correlation_threshold CORRELATION_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT       The folder for storing results
  --imaging IMAGING     The file contains imaging data, the format is RData
  --clinical_file CLINICAL_FILE
                        The file contains clinical variables, the format is
                        csv
  --id_file ID_FILE     The file contains subject id, the format is RData
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