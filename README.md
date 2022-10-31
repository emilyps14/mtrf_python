# mtrf_python

Code associated with the paper
"Latent neural dynamics encode temporal context in speech"

Emily P Stephen, Yuanning Li, Sean Metzger, Yulia Oganian, Edward F Chang

## Package setup.  

### 1. Install required packages

Create a conda environment with the required packages using:
```
conda env create -f environment.yml
conda activate mtrf_python
```

To install the package and have a copy of the code to edit locally, navigate to where you would like to store the package code in your terminal. Clone the package and then install:
```
git clone https://github.com/emilyps14/mtrf_python.git
python -m pip install -e mtrf_python
```

If you just want to use the package as is but still want to install it in your own Python environment, use the following command instead:
```
python -m pip install git+https://github.com/emiyps14/mtrf_python.git
```
 
 
 ### 2. Update the configuration file
 
The `mtrf_config.json` file defines the base path variables that are referenced in the code. We've chosen to use a configuration file as this is a simple way to keep the code general enough but allow the user to change some paths that may be specific to them. The file should be stored at `~/.config/mtrf_config.json`. To run the code from the combined subject dataframe, the only path needed is the subjects directory, e.g.
```
{
  "subjects_dir": "/path/to/subjects/"
}
```

## Run the code

### Necessary data
There is a dataset to accompany this repo, containing the following folders/files:
+ subjects/
    + EC36 [For Single Subject Summary in Supplementary Material]
        + EC36.json
        + EC36_HilbAA_70to150_8band_out_resp_log.pkl
        + EC36_lh_pial_lateral_2dSurf_grid.pkl
    + Hamilton_Agg_LH_no63_131_143/
        + Hamilton_Agg_LH_no63_131_143.json [configuration from prepare_combined_subject_df_script.py]
        + Hamilton_Agg_LH_no63_131_143_HilbAA_70to150_8band_out_resp_log.pkl [data file created by prepare_combined_subject_df_script.py]
        + regression_SO1_PR_phnfeats_750msdelays/
            + Hamilton_Agg_LH_no63_131_143_lh_pial_lateral_2dSurf.pkl [electrode and surface information for plotting, from prepare_2d_surface_script.py]
    + sentence_text.mat [text for the TIMIT sentences]
    + out_sentence_details_timit_all_loudness.mat [feature information for each TIMIT sentence]
    + mel_centerF.mat [center frequencies for the mel spectrograms]

### Running the cross-validated regression models
The code to identify speech-responsive electrodes and to run the cross-validated 
regression models (OLS, Regression, and iRRR) is in the file (After running this script, the size of the subject folder will be about 84.5 Gb. After running collect_cv_results.py, you can delete the folder cv-10fold to reduce space): 
+ projectfiles/run_cv_750msdelays.py.

Code to regenerate figures from the manuscript is in the files:
+ projectfiles/fig1_performance.py
+ projectfiles/fig2_SO_and_PR_components.py
+ projectfiles/fig3_rotational_states.py
+ projectfiles/fig4_decode_latency_from_state.py
+ projectfiles/fig5_scaffolding.py
+ projectfiles/figS1_pca.py
+ projectfiles/figS2_EC36_summary.py