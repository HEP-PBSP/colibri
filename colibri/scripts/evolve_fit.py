"""
A wrapper around n3fit/scripts/evolven3fit.py.

"""
import os
from n3fit.scripts.evolven3fit import main as evolven3fit_main

def main():
    """
    Before running `evolven3fit` from n3fit/scripts/evolven3fit.py,
    creates a symlink called `nnfit` to replicas folder.
    """
    # Create a symlink to the replicas folder
    os.symlink('replicas', 'nnfit')
    
    # Import and run the main function from evolven3fit
    
    evolven3fit_main()
    # Remove the symlink after running
    os.remove('nnfit')
