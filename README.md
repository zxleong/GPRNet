# GPRNet - GPR Inversion Using Deep Learning

Update 1/1/2022: This repo is now updated to reflect the figures and results that are produced in the paper. If you downloaded the files before 1/1/2022, kindly re-download again.

Note: This repository contains only codes. To reproduce the figures at a local machine, please download all data sets here: https://bit.ly/36hDrRj (The password is *a u t o m a t e G P R* , without spaces)

#### Key Points from paper:
- We propose a deep learning-based EM velocity inversion for GPR zero-offset data
- Tests on synthetic examples show accurate velocity inversion results
- Applications to field data yield predictions that agree with the velocity models derived from previous physics-based inversion studies

#### GPRNet Architecture comment:
GPRNet architecture is found at DLcodes/GPRNet.py
- Essentially, it's an encoder-decoder based Convolutional Neural Network (CNN).
- The framework is designed based on the on DeepLabV3 architecture(https://arxiv.org/abs/1706.05587)

#### **Synthetic EM Velocity Inversion**
##### 1D Scenario:-

1. Run *1-Generate_1D_models.py*
   - Generates 10,000 random velocity profiles
   - Creates these files:
     - Synthetic/Data/1D/ep.mat (dielectric permittivity for FDTD simulation to obtain GPR data)
     - Synthetic/Data/1D/veltd.npy (raw velocity files)

2. Run *2-FD_GPR_sim.m*
   - Perform FDTD on ep.mat to create raw 1D GPR gathers
   - Creates Synthetic/Data/1D/fdrawgather.mat

3. Run *3-Process_1D_GPRtraces.py*
   - Removes first arrivals and remove any data pair that has NaN values
   - Creates these files:
     - Synthetic/Data/1D/xTrain_gathers.npy
     - Synthetic/Data/1D/yTrain_vels.npy

4. Run *4-Data_Loader.py*
   - Split data into training, testing, and validation
   - Creates these files:
     - Synthetic/Data/1D/ForDL/Synthetic_Xtrain_1d.npy
     - Synthetic/Data/1D/ForDL/Synthetic_ytrain_1d.npy
     - Synthetic/Data/1D/ForDL/Synthetic_yvalid_1d.npy
     - Synthetic/Data/1D/ForDL/Synthetic_xvalid_1d.npy
     - Synthetic/Data/1D/ForDL/Synthetic_Xtest_1d.npy
     - Synthetic/Data/1D/ForDL/Synthetic_yTrue_1d.npy

5. Run *5-Synthetic_1D_DL_Training.py*
   - Trains GPR-Velocity
   - Creates these files:
     - Synthetic/Weights/weight_GPRNet_n16k20.h5
     - Synthetic/Weights/weight_GPRNet_n16k20.csv

6. Run *6-Synthetic_1D_DL_Prediction.py*
   - Applies trained weights to testing data set
   - Creates these files:
     - Synthetic/Data/1D/ForDL/Synthetic_ypred_1D.npy

7. Run *7-Synthetic_1D.ipynb* (jupyter notebook)
   - **Reproduces Figure 3 and Figure 4**

##### 2D Scenario :- 

8. Run *8-Process_2D_models.py*
   - Synthetic/Data/2D/yTrue2D_vel_dd.npy is the given 2D velocity model (in m)
   - This script converts the velocity model into time depth domain and to dielectric permittivity to be used for 2D common-offset GPR FDTD simulation
   - Creates these files:
     - Synthetic/Data/2D/yTrue2D_ep.mat; (for FDTD simulation)
     - Synthetic/Data/2D/yTrue2D_vel_td.npy; (for ground truthing predicted velocity model)

9. Run *9-FD_sim_2Dtestingmodel.m*
   - Simulates GPR data (common-offset)
   - Creates Synthetic/Data/2D/fdraw_2D.mat

10. Run *10-Synthetic_2D.ipynb* (jupyter notebook) - Part 1
    - Processes and predicts velocity model from fdraw_2D.mat
    - Creates these files:
      - Synthetic/Data/2D/ypred2D.npy
      - Synthetic/Data/2D/ep_ypred2D.mat (used for forward data to see data matching)

11. Run *11-FD_sim_testingmodel_ypred.m*
    - Creates forward data from prediction (ypred2D.npy)
    - Creates Synthetic/Data/2D/fdraw_predicted_data.mat

12. Run *12-Synthetic_2D.ipynb* (jupyter notebook) - Part2 
    - **Reproduces Figure 5 and 6**

#### **Field Application**

13. Run *13-generate_vel.py*
    - Creates these files:
      - Field/Data/ep.mat
      - Field/Data/veltd_raw.mat

14. Run *14-GPR_sim.m*
    - Simulates 50,000 GPR traces (this is a large job, might want to split this into a few parts)
    - Creates these files:
      - Field/Data/fdrawgathers.mat (intermediate files, skipping upload)

15. Run *15-process_GPR_Vel_part1.m*
    - Preprocess GPR data and Velocity
    - Creates these files: (intermediate files, skipping upload)
      - Field/Data/AllRawGathers.mat 
      - Field/Data/veltd_raw_corr.mat 

16. Run *16-process_GPR_Vel_part2.py*
    - Create data and velocity for GPRNet training
    - Augmentation of data set takes place here
    - Creates these files: (intermediate files, skipping upload)
      - Field/Data/ForDL/GPRData.npy
      - Field/Data/ForDL/Vel.npy

17. Run *17-field_Data_Loader.py*
    - Splits data set into training, testing and validation
    - Creates these files:
      - Field/Data/ForDL/field_X_train.npy
      - Field/Data/ForDL/field_X_valid.npy
      - Field/Data/ForDL/field_X_test.npy
      - Field/Data/ForDL/field_y_train.npy
      - Field/Data/ForDL/field_y_valid.npy
      - Field/Data/ForDL/field_y_true.npy

18. Run *18-Field_DL_Training.py*
    - Trains GPR data and Velocity 
    - Creates these files:
      - Field/Weights/weight_GPRNet_n32k10.h5
      - Field/Weights/weight_GPRNet_n32k10.csv

19. Run *19-extract_codata.m*
    - Reads wurtsmith_line1.sgy field data and extracts common-offset data
    - Creates Field/Data/rawfielddata/codata.mat

20. Run *20-Field_Application.ipynb (jupyter notebook)*
    - **Reproduces Figure 7, 8, 9, 10**
    - Applies trained weights to common-offset data to obtain field prediction
    - Follow instructions inside notebook to simulate forward data based on field prediction
    - Creates these files:
      - Field/Data/pcsfielddata/ProcessedFieldData_rev.npy
      - Field/Data/prediction/FieldPrediction_rev.npy
      - Field/Data/prediction/ep_FieldPrediction_rev.mat
      - Field/Data/prediction/rawgather_Stacked1D_fieldprediction_rev.mat (this is created in 20a-FD_pred.m)
      - Field/Data/prediction/ForwDataFrPred_rev.npy

