.. _scitas_guide:

A Guide for Scitas
==================

The goal of this article is to not be a complete how to guide, but to provide a collection of useful links and explanations for how to do common tasks using Scitas.
This is geared towards an ML practicioner looking to run experiments on GPU's in Scitas.
`Scitas's docs <https://scitas-data.epfl.ch/kb>`_ are a great starting place but there was still a learning curve that I hope to make a bit quicker.
You have to be logged in to the VPN to read the docs.

.. important::

    Using Scitas adds complexity and takes some investment to use effectively. I would only move to it should there be an overwhelming need to scale up and use GPU's.

Overview
--------
Scitas is the high-performance computing cluster at EPFL. There are 3 clusters Fidis, Helvetios, and Izar.
To read more about the hardware look `here <https://www.epfl.ch/research/facilities/scitas/hardware/izar/>`_.
Since we are interested in using GPU's we will focus on the Izar cluster.

Follow the instructions for `connecting <https://scitas-data.epfl.ch/confluence/display/DOC/Connecting+to+the+clusters>`_, and `first steps <https://scitas-data.epfl.ch/confluence/display/DOC/First+steps+on+the+clusters>`_.
The `file management system <https://scitas-data.epfl.ch/confluence/display/DOC/File+systems>`_ is described well.
Personally, I stored information as follows:

* HOME - source code, launch files
* SCRATCH - saved outputs of runs (tensorboard output, special saved data, model snapshots)

I would then read `How to use the cluster <https://scitas-data.epfl.ch/confluence/display/DOC/Using+the+clusters#Usingtheclusters-TheDebugPartition>`_.
This is a must read as you start to get up and running as it contains the main commands and explanations of submission scripts.

The next steps that should be followed are:

1. Set up a venv
2. Connect Pycharm by SFTP and load your files
3. Try to launch a test run
4. Debug and try again

.. note::

    You may need to refactor your code so that files are saved and written in the correct location.
    I also found it useful to use .yml files as config files that stored hyperparameters and I could launch runs by referencing the correct .yml file.

Setting up venv
------------------------

.. code-block:: console

    module purge
    module load gcc/8.4.0 python/3.7.7
    virtualenv -p python3 --system-site-packages <venv name>
    source <venv name>/bin/activate

    pip install tensorboard

Now your venv is activated you can install packages like normal using pip.

Setting up Pycharm
------------------
Pycharm has two tools that allow a connection with the remote server.

First, we can set up a remote interpreter to run files on the `servers interpreter <https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html#ssh>`_ through SSH.

The second and more useful is establishing an `SFTP connection <https://www.jetbrains.com/help/pycharm/creating-a-remote-server-configuration.html>`_.
This allows us to sync our local files with the files on the server.
With this we can write code and change files locally and then when ready push these files to the server to then be deployed on the gpu.
Be careful to specify which files you want to sync.
If you have data files it might try to upload and download a lot of data, so it is best to not include them in the sync.
In this way it is similar to github as you only want to upload content and not data.

.. note::

    When setting up a local enviornment (venv/conda) I found it useful to mimic the environment available on the server.
    For example izar uses python 3.7.7 so this is what I installed into my local environment.
    Otherwise when you send your files to the server and there is a mismatch between packages it can cause a lot of headache.


Useful Slurm Commands
---------------------
`Slurm Documentation <https://slurm.schedmd.com/documentation.html>`_

`Scitas Documentation - Running Jobs with SLURM <https://scitas-data.epfl.ch/confluence/display/DOC/Using+the+clusters#Usingtheclusters-RunningjobswithSLURM>`_

.. code-block:: console

    module purge
    module load gcc/8.4.0 python/3.7.7

Sequence of commands to clear all modules and then load tge GCC compiler and python 3.7.7.
This will make it so you can run pip and download packages that are necessary.

.. code-block:: console

    sbatch  <path to .sh file>

Launch a job based on a bash file that specifies the run parameters.

.. code-block:: console

    squeue -alv --me
    squeue -alv --user <username>

This will show any currently running jobs associated to your username

.. code-block:: console

    scancel <jobid>

Cancel a job.

Useful Bash Commands
--------------------
Having a working knowledge of bash commands will be very useful as you can only interact with the server through the command line.
Here is a `cheatsheet <https://github.com/RehanSaeed/Bash-Cheat-Sheet>`_ I used to remind myself of commands.
A few useful commands I found myself using were:

.. code-block:: console

    man <keyword>

This will print out the documentation for any bash command.
Allows you to read for yourself and understand what a command is doing.

.. code-block:: console

    find . -name "<search criteria>"
    find . -name "<search criteria>" -delete

This searches your current folder and below it for a search criteria and shows the matches.
If you are happy with what was matched you can use this to delete files which match this criteria.
I found this useful in cleaning up and deleting old scratch runs.

.. code-block:: console

    history|grep <search criteria>
    !<line number>

Allows you to search your command history and then execute a command based on the line number.
Useful to not have to re-type a long command.

.. code-block:: console

    watch -n 1 tail <filepath>

This will print the last 10 lines of a file and update every 1 second.
Useful to watch the slurm output to see if your code is running as expected or if it threw an error.


Bash Scripts
------------
To send jobs to slurm, it is very useful to use a bash script.
I used this `cheatsheet <https://devhints.io/bash>`_ to learn more about how to write bash scripts.
Scitas documentation also has a `github with examples <https://c4science.ch/source/scitas-examples/>`_.

Some bash scripts that I used to launch runs can be found in my github.

* :doc:`slurm_launch/gpu_run`
    - Launch a single GPU run
* :doc:`slurm_launch/debug_run`
    - Launch a single GPU run in debug mode for testing purposes
* :doc:`slurm_launch/launch_jupyter`
    - Launch a jupyter notebook
* :doc:`slurm_launch/normal_run`
    - This was a special case where I didn't use a gpu and had to launch multiple runs on the deneb cluster.


Using Tensorboard (Port Forwarding)
-----------------------------------
I found it useful to inspect the tensorboard outputs as training was occuring on the gpu.

Start a new terminal window and then open up a SSH tunnel with portforwarding.
Then launch tensorboard with the corrrect port on the server.
Don't forget to activate the venv that you need and/or load an modules.
In the following I portforward from 16008 to 6008.

.. code-block:: console

    ssh -L localhost:16008:localhost:6008 <username>@izar.epfl.ch
    source activate <venv>/bin/activate
    tensorboard --logdir /scratch/izar/<username>/ --port=16008

Back on your local computer paste the tensorboard path with the correct port into a web browser.
http://localhost:<6008>/

For more information:

`Stack overflow on running tensorboard on remote server <https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server>`_

`Potential error with portbinding <https://askubuntu.com/questions/447820/ssh-l-error-bind-address-already-in-use>`_

Using JupyterNotebooks (Port Forwarding)
----------------------------------------
This takes a bit more effort but I found it useful to use Jupyter Notebooks after I had trained my models and I wanted to test them and save various data from the networks.
`The How to use Jupyter and Tensorflow on Izar <https://scitas-data.epfl.ch/confluence/display/DOC/How+to+use+Jupyter+and+Tensorflow+on+Izar>`_ covers this nicely.

.. note::

    If you want to use a jupyter notebook in your home directory, you have to launch from your home directory.
    Any files that get written from your jupyter will write to your home directory unless you specify a path.
    In my case I wanted to write files to scratch so I would specify the output path in the jupyter notebook.




