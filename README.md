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


