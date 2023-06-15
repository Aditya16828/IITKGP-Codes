from sippy import system_identification
from sippy import functionset as fset
from sippy import functionsetSIM as fsetSIM
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

class Model:
    inputData = np.array([])
    outputData = np.array([])
    METHOD = ['N4SID','CVA','MOESP','PARSIM-S','PARSIM-P','PARSIM-K']

    def __init__(self, input, output) -> None:
        self.inputData = input
        self.outputData = output
    
    def addNoise(self, size, variances):
        noise = fset.white_noise_var(size, variances)
        netOutput = self.outputData + noise
        return netOutput
    
    def model_withoutNoise(self):
        sysId = system_identification(self.outputData, self.inputData, self.METHOD[1])
        x_id, y_id = fsetSIM.SS_lsim_process_form(sysId.A, sysId.B, sysId.C, sysId.D, self.inputData, sysId.x0)
        return [sysId, x_id, y_id]
    
    def model_withNoise(self, size, variances):
        output = self.addNoise(size, variances)
        sysId = system_identification(output, self.inputData, self.METHOD[1])
        x_id, y_id = fsetSIM.SS_lsim_process_form(sysId.A, sysId.B, sysId.C, sysId.D, self.inputData, sysId.x0)
        return [sysId, x_id, y_id, output]
    
    def predict(self, sysModel, inputs):
        xId, yId = fsetSIM.SS_lsim_process_form(sysModel.A, sysModel.B, sysModel.C, sysModel.D, inputs, sysModel.x0)
        return [xId, yId]