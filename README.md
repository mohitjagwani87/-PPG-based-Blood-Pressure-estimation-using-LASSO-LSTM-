PPG-Based Blood Pressure Estimation using Machine Learning / Deep Learning
                                                                     #Overview

This project focuses on cuff-less blood pressure (BP) estimation using photoplethysmography (PPG) signals.
The goal is to estimate Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP) from PPG waveforms through a robust pipeline of signal preprocessing, cycle segmentation, feature extraction, and machine learning regression.

This implementation uses a dataset collected from AIMS Hospital, containing 3-channel biosignals (ECG, Respiration, PPG) from 46 subjects, each with 268,289 samples.
Subject-level SBP and DBP readings were recorded in mmHg and used as target variables.

                                                                        # Methodology
1. Data Acquisition
Each .acq file contains three channels:
Channel 1: ECG
Channel 2: Respiration
Channel 3: PPG
Subject metadata (SBP/DBP, age, etc.) was stored in a separate CSV file (Mater_Dataset_SBP_DBP.csv).

2. Signal Preprocessing
Steps performed:
Segmentation: The PPG signal was divided into 10-second segments (1250 samples, 50% overlap).
Detrending: A polynomial-based Subtract Polynomial Approximation (SPA) method was used to remove baseline drift.
Smoothing: Savitzky–Golay filtering (window_length=51, polyorder=3) for noise suppression.

3. Cycle Detection
Detected peaks and valleys using scipy.signal.find_peaks.
Cycles were validated by checking inter-peak intervals between 0.25 s to 1.25 s, ensuring physiologically realistic PPG cycles.
Each valid cycle was extracted from the detrended signal for feature computation.
SNR Filtering: Segments with SNR ≥ 3 dB (computed using Welch’s method) were retained for further analysis.

4. Feature Extraction

For every valid cycle, 54 handcrafted features were extracted, categorized as:

Category	Description	Count
Morphological	Shape, symmetry, and curvature (e.g., SD, Skewness, Kurtosis, K-value)	4
Gradient-based	1st & 2nd derivatives during systolic/diastolic phases	8
Time-based	Cycle length, systolic/diastolic time ratio, etc.	4
Amplitude-based	AC, DC, systolic and diastolic areas, reflection index	6
Width ratios (PWx, SWx, DWx, PWRx)	Pulse widths at 10–90% amplitude levels	28
HRV features	Derived from inter-peak intervals (RMSSD, SDNN, LF/HF)	4
Each cycle generated a 54-dimensional feature vector.

5. Feature Selection (Filter-Wrapper Approach)
Filter stage: Mutual Information (MI) scores computed between each feature and SBP target.
Wrapper stage: Highly correlated features were pruned using correlation threshold r > 0.9.
The final 15 non-redundant, high-relevance features were retained.
Top Selected Features:
AC, PW66, PW75, SDNN, DSmax, DAmax, ASmax, AAmax, DW50, PW90, SDSD, DAmean, PWR75, K_value, SW90

6. Model Development
Two modeling approaches were tested:
LASSO-LSTM (Deep Learning) – as per the IEEE reference paper (for research version)
Random Forest Regressor – implemented here for robust interpretability and performance on the AIMS dataset.
The model was trained separately for SBP and DBP using subject-level target mapping.

7. Training and Evaluation
Subject-wise train-test split (80:20) ensuring no subject overlap.
Features were normalized using StandardScaler.
Metrics used:
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Coefficient of Determination (R²)------ NEGATIVE for nonlinear relationship model
Explained Variance (EVS)

Hnece, check this out!
         ┌────────────────────────┐
         │ Raw PPG (from .acq)    │
         └────────────┬───────────┘
                      │
                      ▼
           ┌────────────────────┐
           │ Preprocessing       │
           │ - Detrending (SPA)  │
           │ - Smoothing (SG)    │
           │ - SNR ≥ 3 filtering │
           └─────────┬───────────┘
                     │
                     ▼
           ┌────────────────────┐
           │ Cycle Detection     │
           │ - Peak/Valley       │
           │ - Inter-peak check  │
           └─────────┬───────────┘
                     │
                     ▼
           ┌────────────────────┐
           │ Feature Extraction  │
           │ - 54 handcrafted    │
           │   features          │
           └─────────┬───────────┘
                     │
                     ▼
           ┌────────────────────┐
           │ Feature Selection   │
           │ - MI + Correlation  │
           │ - Final 15 features │
           └─────────┬───────────┘
                     │
                     ▼
           ┌────────────────────┐
           │ ML Model (RF)      │
           │ - Train/Test split │
           │ - Evaluate Metrics │
           └────────────────────┘
<img width="1423" height="1040" alt="EdrawMax-AI-diagram (2) (1)" src="https://github.com/user-attachments/assets/a253004c-e017-4be0-b843-b05074631b3d" />
                                                     
                                                                         #Results
Parameter	SBP (mmHg)	DBP (mmHg)
PERFORMANCE METRICS FOR SBP AND DBP ESTIMATION
Metric                SBP    DBP
MAE (mmHg)            4.00   4.60
MSE                   29.86  32.75
RMSE (mmHg)           5.46   5.72
Median AE (mmHg)      2.80   3.69
Max Error (mmHg)      19.85  18.68
These errors are in true clinical units (mmHg) since the targets (SBP/DBP) were directly taken from the metadata without normalization.
The results indicate that the model achieves consistent and clinically relevant accuracy, with lower error for DBP and higher variance for SBP (consistent with literature trends).
<img width="244" height="97" alt="image" src="https://github.com/user-attachments/assets/8b179992-e1fc-4074-9af3-c2c12cd1e2ba" />

                                                                        #Key Contributions

Developed a complete end-to-end cuff-less BP estimation pipeline using PPG signals.
Implemented signal-quality-based filtering (SNR ≥ 3 dB) and SPA-based detrending for robust preprocessing.
Extracted 54 handcrafted features, achieving 15 key non-redundant features strongly correlated with BP.
Built a subject-independent Random Forest model for accurate SBP/DBP prediction.
Achieved clinically interpretable results with mean errors within acceptable BP estimation range.

├── data/
│   ├── acq_data_master/         # .acq files for each subject
│   ├── Mater_Dataset_SBP_DBP.csv
├── preprocessing/
│   ├── detrend_spa.py
│   ├── snr_filter.py
│   ├── cycle_segmentation.py
├── features/
│   ├── extract_features.py
│   ├── feature_selection.py
├── models/
│   ├── random_forest_regressor.py
│   ├── lasso_lstm_model.py
├── results/
│   ├── sbp_dbp_predictions.csv
│   ├── metrics_summary.txt
└── main_pipeline.py
