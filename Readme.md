# Directory Structure
1. src/   Source code for the model training process
    * src/poc.txt Training data
    * poc_ner_rebertap_arg.ipynb Training script
2. demo_data/    Test input files for PoC information preprocessing
    * *nlp.txt Test input file
    * *aspects.json Test output file
3. trained_model/    Model archive for PoC information preprocessing
4. POCAspectExtraction.py   PoC information preprocessing test script

# Environment Dependencies
1. conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
2. pip install transformers
3. pip install datasets

# Usage (Testing Method)
1. Run the test script: `python POCAspectExtraction.py`
2. Review the *aspects.json file to verify that the model correctly extracts key aspects from the PoC report.
