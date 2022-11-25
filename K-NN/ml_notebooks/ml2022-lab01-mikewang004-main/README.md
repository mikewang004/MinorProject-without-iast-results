# MLCourse-LU-Lab1
This lab contains a tutorial of the key tools to be used during the remainder of this course, as well as a few (ungraded) exercises.
Although this lab is not graded we strongly recommend you to follow this lab, we will be explaining important concepts that you need to understand to successfully complete and submit the graded assignments.

## How to start
* Start by cloning this repository to your computer. 
	We will be using GitHub for both handing out the assignments and for submitting and grading them.
	This means that a basic understanding of git is required. 
	To completely explain git is out of scope for this lab but we encourage everybody who isn't familiar to try to follow some tutorials online, understanding git is extremely useful for any programmer.
	
	We will explain the steps to start the assignment using the GitBash terminal, feel to skip this section and use your own way to interact with Git.
	Start by downloading git on your machine if you haven't already: https://git-scm.com/downloads
	
	To start this lab you need to be able to clone the repository, this means getting the code locally on your machine.
	Open Git bash terminal and navigate to the location on your computer where you want the lab to be downloaded to.
	Execute the following command: 
	
	> git clone https://github.com/MLCourse-LU/MLCourse-LU-Lab1.git
	
	Note that you might be asked for authentication if so consult the GitHub help pages and follow the instructions there.
	
* You can do the assignment on your own computer via a jupyter notebook, or use Google Colab.
    1. To run it **locally**, make sure that you have Python 3.6 or higher installed. Try running `jupyter notebook` from the command line. 
       
       * If it works, open `Lab1.ipynb` and continue with the explanations and tasks there.
       
        * If that doesn't work, you first have to install Jupyter notebook. You may also have to install other packages: `numpy`, `pandas`, `matplotlib` and `pytest`. You can do this either via pip or anaconda.
    2. To use **Google Colab**, upload `Lab1.ipynb` and proceed with the notebook from there. 

If you have difficulty installing a needed package, we recommend either asking for help or switching to Colab. In Colab, to install a package, you can run a cell with the command:

`!pip install (name of package)`

The `!` will send the command out of the notebook, to the command line of Google Colab.
