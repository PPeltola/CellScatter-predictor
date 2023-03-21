# CellScatter Predictor

MVP implementation of the tool developed by the CellScatter group for the University of Helsinki Data Science Project (spring -23) course. Some project notebooks used to develop the machine learning models used in the tool can be found [here](https://github.com/K123AsJ0k1/CellScatter).


## Installation
 
 TODO


## How to use

The module contains the ```CellScatterPredictor``` class, which should be imported: 

```from cellscatter_predictor import CellScatterPredictor```


To make predictions, create an instance of the class and call the ```.predict``` function:
![](https://github.com/PPeltola/CellScatter-predictor/blob/main/documentation/images/predict1.png)

The data given as the parameter should be a one-dimensional vector of 1000 values, at least python lists, NumPy arrays and pandas Series work fine.


The ```.predict``` function returns a dictionary of the predicted values, where the keys are ```density```, ```thickness``` and ```APL```:

![](https://github.com/PPeltola/CellScatter-predictor/blob/main/documentation/images/preds1.png)

The ```thickness``` and ```APL``` have single-value predictions, but the ```density``` values contain 200 (x, y) pairs from the predicted density:

![](https://github.com/PPeltola/CellScatter-predictor/blob/main/documentation/images/density1.png)
