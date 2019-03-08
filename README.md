# Distinguishing Narration and Speech in Literary Fiction Dialogues

Model to identify narration inside lines of direct speech.

To run training and testing: 
    
    python3 main.py
    
    
    
Data is located in ./data/

`data_wgold.txt` contains the lines with annotations indicating narration (inside `<NC>` tags).

`tagged_data.conll` contains the lines parsed with `efselab` (https://github.com/robertostling/efselab).

`data.pickle`: contains the parsed lines with labels.

