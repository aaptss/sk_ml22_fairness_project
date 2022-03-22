# Adaptive Sensitive Reweighting
This project is python adaptation of otiginal [Adaptive Sensitive Reweighting](https://github.com/maniospas/AdaptiveFairness).

Original description:
> Implementation of an algorithmic framework for achieving optimal fairness-accuracy trade-offs.
> This is a MATLAB project which utilizes adaptive sensitive reweighting to produce fair classifications.

## Classifiers
Folder *+classifiers* includes fairness-aware method, alongside some similar experimental approaches.

## Datasets
Folder *+dataImport* includes Adult, Bank, Census and COMPAS datasets and two synthetic disparate mistreatment datasets.
Datasets can be imported using the respective loader (matlab-function) from this folder (e.g. *dataImport.importAdultData()*).

## Requirements
There are two ways to launch classifier:
* using **Matlab** software
* via terminal using **Python3** and **Octave**

### Matlab Requirements
* Matlab R2012b+

### Python3 and Octave Requirements
* Python3.10 (tested, versions below might work too)
* Octave6.4 (tested, versions below might work too)

Octave for Python installation:
```bash
pip install octave-kernel
```

Also you need to [install Octave](https://wiki.octave.org/Category:Installation) itself.
Below is the giude for Octave for Debian systems (other installation options can be found in the link above):
```bash
sudo apt-get install octave
```
Add packages:
```bash
sudo apt-get install octave-control octave-image octave-io octave-optim octave-signal octave-statistics
```

## Get started

Currently there are 4 loaders:
1. Adult dataset
2. Bank dataset
3. Census dataset
4. Compass dataset

### Matlab
Run `main.m` with one of possible argument:
1. @()dataImport.importAdultData()
2. @()dataImport.importBankData()
3. @()dataImport.importCensusData()
4. @()dataImport.importCompassData()

It will return metrics using classifier and setted number of folds.

To change number of folds: edit **line 4** in `main.m` file

To change the classifier edit **line 8** in `main.m` file. You can choose classifier from *+classifiers* folder.

### Python3 and Octave
Run `run.py` file with `-d` flag which choose dataset:
1. `python3 run.py -d adult`
2. `python3 run.py -d bank`
3. `python3 run.py -d census`
4. `python3 run.py -d compass`

It will return metrics using classifier and setted number of folds.

To change number of folds: edit **line 4** in `main.m` file

To change the classifier edit **line 8** in `main.m` file. You can choose classifier from *+classifiers* folder.
