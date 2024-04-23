## Counterfactual Explanations with Probabilistic Guarantees on their Robustness to Model Change

Paper ID: 1467

***
### Setup
Conda:
```bash
conda env create --name envname --file=environment.yml
```
Pip (tested on Python 3.11.7):
```bash
python -m pip install -r requirements.txt 
```

***
### Usage

Directory structure:   
        `bash_scripts`  
        `configs`  
        `images`  
        `data`   
        `src`   
        config.yml  
        ├── `explainers`  
        ├── `helpers`  
        ├── `models`  
        ├── `paper_utils`  
        ├── betarce.py  
        ├── experimentsv3.py  
        └── robx.py  
  

*Make sure that your python is calling scripts from the root directory level to avoid issues with wrong paths  

Brief description of the scripts:  
- **experimentsv3.py** script contains the information you need to implement end-to-end BetaRCE method in your pipeline.
- **betarce.py** script contains the BetaRCE class which implements BetaRCE method described in the paper.   
- **robx.py** is our implementation of RobX method.  
- `models` folder contains different classifiers (black-box models)
- `explainers` contains implementation of GrowingSpheres and Dice countercactual explanations generation algorithms 
- `helpers` has some utility scripts
- `paper_utils` contains all scripts and notebooks used to process data for the paper and create visualizations 



***
### Reproduce experiments
To run experiments for the paper we utilized our internal compute cluster running on slurm. Therefore, our scritps in `bash_scripts` contain directions for SBATCH commands. But, they can be ran without using slurm.  

To fully reproduce experiments we created a bash script **bash_run_all_experiments.sh**, which generates the exact copy of the data we obtained. Running all experiments is very time-consuming, so we recommend to distribute the computing to multiple nodes if possible.    

After completing jobs created with **bash_run_all_experiments.sh**, you need to run `python src/helpers/process_results.py` in order to make the results ready to be read by notebooks creating visualizations in `src/paper_utils`. Everything is set to work out-of-the box, but if you make some adjustments to paths in scripts running the experiments, then please review the **process_results.py** and make proper adjustments. 

We do not provide data in this anonymous submission because it is about 80MBs, and the limit in EasyChair is set to 50MBs. We also don't make it available online, because we don't want to violate submission rules, which state that:   

*(3) Online material: Links to supplementary material you make available online, even if anonymous, are not permitted. Instead, submit your supplementary material through EasyChair.*  -- from the email from ECAI'24 PC Chairs on 22 April 2024

But, at least, the notebooks provided have their output cells preserved.
