import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from mapie.regression import MapieRegressor
from tensorflow import keras

class CellScatterPredictor:
    def __init__(self,
                 thickness_pickle="thickness.pkl", 
                 apl_pickle="apl.pkl",
                 density_stats_pickle="density_stats.pkl",
                 density_model="MVP_TD_model.h5"):
        
        with open(thickness_pickle, 'rb') as f:
            thickness_data = pickle.load(f)
            self.thi_formfactor_mean = thickness_data['formfactor_mean']
            self.thi_formfactor_std = thickness_data['formfactor_std']
            self.thickness_mean = thickness_data['thickness_mean']
            self.thickness_std = thickness_data['thickness_std']
            self.thickness_model = thickness_data['model']
            
        with open(apl_pickle, 'rb') as f:
            apl_data = pickle.load(f)
            self.apl_formfactor_mean = apl_data['formfactor_mean']
            self.apl_formfactor_std = apl_data['formfactor_std']
            self.apl_mean = apl_data['apl_mean']
            self.apl_std = apl_data['apl_std']
            self.apl_model = apl_data['model']
            
        
        with open(density_stats_pickle, 'rb') as f:
            density_stats = pickle.load(f)
            self.density_formfactor_mean = density_stats['formfactor_mean']
            self.desnity_formfactor_std = density_stats['formfactor_std']
            self.density_y_mean = density_stats['TD_y_mean']
            self.density_y_std = density_stats['TD_y_std']
            self.density_x_std = density_stats['TD_x_std']
            self.density_model = keras.models.load_model(density_model)
        
        self.quantiles = [0.5, 0.25, 0.05]
    
    def _normalize(self, x, mean, std):
        return (x - mean) / std
    
    def _denormalize(self, x, mean, std):
        return (x * std) + mean
    
    def _print_regression_preds(self, feature: str, pred, pis):
        print(f"Predicted {feature}\t {pred:.6f}")
        for i, q in enumerate(self.quantiles):
            perc = int((1 - q) * 100)
            print(f"  {perc}% pred. interval:\t({pis[0][i]:.4f}, {pis[1][i]:.4f})")
        print()
    
    def predict_thickness(self,
                          formfactor, 
                          print_text=True):
        
        pred, pis = self.thickness_model.predict(
            [self._normalize(formfactor, 
                             self.thi_formfactor_mean, 
                             self.thi_formfactor_std)], 
            alpha=self.quantiles
        )
        
        pred = self._denormalize(pred[0], self.thickness_mean, self.thickness_std)
        pis = self._denormalize(pis[0], self.thickness_mean, self.thickness_std)
        
        if print_text:
            self._print_regression_preds("thickness:", pred, pis)
                
        return pred
    
    def predict_apl(self,
                    formfactor, 
                    print_text=True):
        
        pred, pis = self.apl_model.predict(
            [self._normalize(formfactor, 
                             self.apl_formfactor_mean, 
                             self.apl_formfactor_std)], 
            alpha=self.quantiles
        )
        
        pred = self._denormalize(pred[0], self.apl_mean, self.apl_std)
        pis = self._denormalize(pis[0], self.apl_mean, self.apl_std)
        
        if print_text:
            self._print_regression_preds("APL:\t", pred, pis)
                
        return pred
    
    def predict_density(self,
                       formfactor,
                       print_text=True,
                       plot=True,
                       ff_name=None):
        
        pred = self.density_model(
            self._normalize(formfactor,
                            self.density_formfactor_mean,
                            self.density_formfactor_mean)
            .reshape(1, -1)
        )
        
        pred_xs = self._denormalize(pred[0][:200], 0, self.density_x_std)
        pred_ys = self._denormalize(pred[0][200:], self.density_y_mean, self.density_y_std) + 333.3
        
        if plot:
            plt.plot(pred_xs, pred_ys, color='b')
            plt.xlim(pred_xs[0], pred_xs[199])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Predicted density of ' + str(ff_name)) if ff_name else plt.title('Predicted density')
            plt.show()
            
        return list(zip(pred_xs.numpy(), pred_ys.numpy()))
    
    def predict(self,
                formfactor,
                print_text=True,
                plot_density=True,
                ff_name=None):
        
        density_pred = self.predict_density(
            formfactor,
            print_text,
            plot_density,
            ff_name
        )
        
        thickness_pred = self.predict_thickness(
            formfactor,
            print_text
        )
        
        apl_pred = self.predict_apl(
            formfactor,
            print_text
        )
        
        preds = {
            'density': density_pred,
            'thickness': thickness_pred,
            'APL': apl_pred
        }
        
        return preds
