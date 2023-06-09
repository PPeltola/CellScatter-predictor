# CellScatter Predictor

MVP implementation of the tool developed by the CellScatter group for the University of Helsinki Data Science Project (spring -23) course. Some project notebooks used to develop the machine learning models used in the tool can be found [here](https://github.com/K123AsJ0k1/CellScatter). The goal is to eventually make this tool available on PyPI as well, but the current MVP version is only available here as a download.


## Installation
 
Clone the repository from GitHub. The dependencies can be installed using pip by running ```pip install -r requirements.txt``` in the project root folder. It's **very much recommended** to create a Python virtual environment where you install the denpendencies and the module, since the dependencies include very large packages such as TensorFlow.


## Running the example notebook

The repository includes two Jupyter notebook with example code: one showing the example use cases shown below and one with code for handling the experimental formfactor data.

To try out the example notebooks, simply run ```jupyter-lab``` (from the virtual environment) after installing the requirements in your virtual environment. Jupyterlab is included in the installed requirements. The notebooks and all the data files used in the notebooks are found in the ```notebooks``` folder.


## How to use

The module contains the ```CellScatterPredictor``` class, which should be imported: 

```from cellscatter_predictor import CellScatterPredictor```


To make predictions, create an instance of the class and call the ```.predict``` function:
![](https://github.com/PPeltola/CellScatter-predictor/blob/main/documentation/images/predict1.png)

The data given as the parameter should be a one-dimensional vector of 1000 values, at least regular Python lists, NumPy arrays and pandas Series work fine.


The ```.predict``` function returns a dictionary of the predicted values, where the keys are ```density```, ```thickness``` and ```APL```:

![](https://github.com/PPeltola/CellScatter-predictor/blob/main/documentation/images/preds1.png)

The ```thickness``` and ```APL``` have single-value predictions, but the ```density``` values contain 200 (x, y) pairs from the predicted density:

![](https://github.com/PPeltola/CellScatter-predictor/blob/main/documentation/images/density1.png)

The different properties can also be predicted separately, and the printing/plotting during prediction can be silenced by using the ```print_text=False``` and ```plot=False``` parameters.

![](https://github.com/PPeltola/CellScatter-predictor/blob/main/documentation/images/preds2.png)

These silencing parameters can be given to the ```.predict``` function as well:

![](https://github.com/PPeltola/CellScatter-predictor/blob/main/documentation/images/predict2.png)
