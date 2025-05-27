## DualSHAP
### Usage

```
python iteration.py --csv_dir /path/to/your/tabular/data/csv/files --num_classes 2 (# of target)
```

`--csv_dir`: Path to the directory containing your tabular data csv files.

`--num_classes`: Number of target classes for the classification task. 2 for binary classification, 3 for three-class classification, and so on.

After running the command, the script will generate feature importance visualizations using both SHAP and DualSHAP
