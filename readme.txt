
Synthetic EM Velocity Inversion

Run codes from 

GPRNet architecture is found at DLcodes/GPRNet.py

1D Scenario:-

1. Run 1-Generate_1D_models.py
   - Generates 10,000 random velocity profiles
   - Creates these files:
     - Synthetic/Data/1D/ep.mat (dielectric permittivity for FDTD simulation to obtain GPR data)
     - Synthetic/Data/1D/veltd.npy (raw velocity files)

Run 2-FD_GPR_sim.m
Perform FDTD on ep.mat to create raw 1D GPR gathers
Creates Synthetic/Data/1D/fdrawgather.mat

Run 3-Process_1D_GPRtraces.py
Removes first arrivals and remove any data pair that has NaN values
Creates these files:
Synthetic/Data/1D/xTrain_gathers.npy
Synthetic/Data/1D/yTrain_vels.npy

Run 4-Data_Loader.py
Split data into training, testing, and validation
Creates these files:
Synthetic/Data/1D/ForDL/Synthetic_Xtrain_1d.npy
Synthetic/Data/1D/ForDL/Synthetic_ytrain_1d.npy
Synthetic/Data/1D/ForDL/Synthetic_yvalid_1d.npy
Synthetic/Data/1D/ForDL/Synthetic_xvalid_1d.npy
Synthetic/Data/1D/ForDL/Synthetic_Xtest_1d.npy
Synthetic/Data/1D/ForDL/Synthetic_yTrue_1d.npy

Run 5-Synthetic_1D_DL_Training.py
Trains GPR-Velocity
Creates these files:
Synthetic/Weights/weight_GPRNet_n16k20.h5
Synthetic/Weights/weight_GPRNet_n16k20.csv

Run 6-Synthetic_1D_DL_Prediction.py
Applies trained weights to testing data set
Creates these files:
Synthetic/Data/1D/ForDL/Synthetic_ypred_1D.npy

Run 7-Synthetic_1D.ipynb (jupyter notebook)
Reproduces Figure 3 and Figure 4

2D Scenario :- 

Run 8-Process_2D_models.py
Synthetic/Data/2D/yTrue2D_vel_dd.npy is the given 2D velocity model (in m)
This script converts the velocity model into time depth domain and to dielectric permittivity to be used for 2D common-offset GPR FDTD simulation
Creates these files:
Synthetic/Data/2D/yTrue2D_ep.mat; (for FDTD simulation)
Synthetic/Data/2D/yTrue2D_vel_td.npy; (for ground truthing predicted velocity model)

Run 9-FD_sim_2Dtestingmodel.m
Simulates GPR data (common-offset)
Creates Synthetic/Data/2D/fdraw_2D.mat

Run 10-Synthetic_2D.ipynb (jupyter notebook) - Part 1
Processes and predicts velocity model from fdraw_2D.mat
Creates these files:
Synthetic/Data/2D/ypred2D.npy
Synthetic/Data/2D/ep_ypred2D.mat (used for forward data to see data matching)

Run 11-FD_sim_testingmodel_ypred.m
Creates forward data from prediction (ypred2D.npy)
Creates Synthetic/Data/2D/fdraw_predicted_data.mat

Run 12-Synthetic_2D.ipynb (jupyter notebook) - Part2 
reproduces Figure 5 and 6

Field Application 

Run codes from Field/

Run 13-generate_vel.py
Creates these files:
Field/Data/ep.mat
Field/Data/veltd_raw.mat

Run 14-GPR_sim.m
Simulates 50,000 GPR traces (this is a large job, might want to split this into a few parts)
Creates these files:
Field/Data/fdrawgathers.mat (this is a large file, so I didn’t include it here)

Run 15-process_GPR_Vel_part1.m
Preprocess GPR data and Velocity
Creates these files: (these intermediate files are large. skipping upload)
Field/Data/AllRawGathers.mat 
Field/Data/veltd_raw_corr.mat 

Run 16-process_GPR_Vel_part2.py
Create data and velocity for GPRNet training
Augmentation of data set takes place here
Creates these files: (these intermediate files are large. skipping upload)
Field/Data/ForDL/GPRData.npy
Field/Data/ForDL/Vel.npy

Run 17-field_Data_Loader.py
Splits data set into training, testing and validation
Creates these files:
Field/Data/ForDL/field_X_train.npy
Field/Data/ForDL/field_X_valid.npy
Field/Data/ForDL/field_X_test.npy
Field/Data/ForDL/field_y_train.npy
Field/Data/ForDL/field_y_valid.npy
Field/Data/ForDL/field_y_true.npy

Run 18-Field_DL_Training.py
Trains GPR data and Velocity 
Creates these files:
Field/Weights/weight_GPRNet_n32k10.h5
Field/Weights/weight_GPRNet_n32k10.csv

Run 19-extract_codata.m
Reads wurtsmith_line1.sgy field data and extracts common-offset data
Creates Field/Data/rawfielddata/codata.mat

Run 20-Field_Application.ipynb (jupyter notebook)
Reproduces Figure 8, 9, 10
Applies trained weights to common-offset data to obtain field prediction
Follow instructions inside notebook to simulate forward data based on field prediction
Creates these files:
Field/Data/pcsfielddata/RawFieldData.npy
Field/Data/pcsfielddata/ProcessedFieldData.npy
Field/Data/prediction/FieldPrediction.npy
Field/Data/prediction/ep_FieldPrediction.mat
Field/Data/prediction/rawgather_Stacked1D_fieldprediction.mat (this is created in 20a-FD_pred.m)
Field/Data/prediction/ForwDataFrPred.npy

Uncertainty Quantification

21-Run Synthetic_UQ.ipynb (jupyter notebook)
Reproduces Figures 11, 12, 13, 14
Simulates uncertainties with regards to dropout
Creates these files:
UQ/Data/Synthetic/all_dropout_predictions_0p05.npy
UQ/Data/Synthetic/all_dropout_predictions_0p1.npy
UQ/Data/Synthetic/all_dropout_predictions_0p2.npy
UQ/Data/Synthetic/all2D_dropout_predictions_example.npy


22-Run Field_UQ.ipynb (jupyter notebook)
Reproduces Figures 15, 16
Produces uncertainties for field prediction
Files created beforehand (read notebook for more details):
UQ/Data/Field/X_test_spec.npy
UQ/Data/Field/y_True_spec.npy
Creates these files:
UQ/Data/Field/ypred_spec.npy
UQ/Data/Field/field_dropout_preds.npy
