# Dynamic Degree Polynomial Neural Network

The [Project Report](/paper/Dynamic_Poly_NN.pdf).

Outputs of tests are saved in the scratch folder.
To see and run plots from the paper, can open and run the [plotting.ipynb](/plotting.ipynb) jupyter notebook. 
This will load data saved in the scratch folder and plot it. 

**To train a model the steps are:**

1) Set up a .yml file with the training hyperparameter
   1) [Examples of configurations](/configs/)
2) Execute `run.py "<FILENAME>.yml"`
3) Output of run will be saved as `/{REPO ROOT}}/scratch/logs/<DATASET USED>/<NETWORKNAME>_<CUSTOM SAVE NAME>_<DATETIME>`
   1) `<DATETIME>.log` - log of outputs logged / printed to command line
   2) `best_model` - snapshot of the best model based on validation accuracy
   3) `events.out.tfevents` - tensorboard output file
   4) `latest_e<EPOCH>_t<CUMULATIVE TIME>` - snapshot of model saved every X epcchs
   5) `r_<YAML NAME>.yml` - copy of yaml file that details the hyperparameters used

Read the documentation regarding [how to use Scitas (EPFL HPC cluster)](/scitas_guide.rst).:

**To train a model using Scitas the steps are:**
1) Read my [documentation](/scitas_guide.rst) and set up files on the SCITAS cluster
2) Set up a bash script. For [examples](slurm_launch) see my repo. The following need to be specified in the bash script:
   1) SCITAS Cluster Parameters
   2) Load appropriate environment packages (Interpreter, Python, Packages)
   3) Runs appropriate python file
4) Run bash script by using slurm command `sbatch <PATH TO BASH SCRIPT>`
