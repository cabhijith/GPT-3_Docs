# Training GPT-3 to learn Harry Potter. 
The aim here is to train GPT-3 on Harry Potter, particularly imitating the writing style of it. 

[Click here](https://github.com/cabhijith/GPT-3_Docs/blob/master/Fine-Tune.md) to learn more about training/inference of fine-tuned models. 

## Training
The data was extracted from PDF copies of Harry Potter. My first approach was to just chuck each page as one element of a list and then convert it to JSONL. Like this: 
```python 
[{"data" : "Page-1"}, 
 {"data": "Page-2"}
 ...
 ]
 ```
 This gave okay results but I thought it was better to have it in the form of a ```text_list``` where one element resembles a prompt and the other its completion. Here's how the data looked: 
 ```python
 [{'data': {'text_list': ["Starting of Page-2",
    "Remaining Part of Page-1"],
   'loss_weights': [0.3, 0.7]}},
 {'data': {'text_list': ['Starting of Page-2',
    "Remaining part of Page-2"],
   'loss_weights': [0.3, 0.7]}}
  ... 
   ]
 ```
 The prompt was selected as a random amount of text from each page (between 20-50%). I also added ```loss_weights``` giving more importance to the completion rather than the prompt. This worked much better than the earlier approach. Command used to train: 
 ```python 
 !openai-ft -t data/train_harry.jsonl --val data/test_harry.jsonl --num-epochs 1 -e ada-abhijith-chandran-ft-c2 -m ada --batch-size 2 --val-batch-size 1 --snapshots-every 15 --num-completions 1 --completions-every 2 -s 0.01 --log-path logs -v 
```
I initially trained it with the default learning rate for 1 epoch but found that the model overfitted a bit. Instead of learning the style it completed prompts using the same characters from Harry Potter. Scaling the learning rate down by 100x worked much better. 

## Testing 
**Prompt**: Standing in the doorway, she stared at the empty hallways. Suddenly 

**Completion**: the lights of her office went out. They were replaced by a blank, dark room. Overhanging the ceiling were thick gray curtains, and the hole in the ceiling's center sparkled with yellowish reflection. She had never seen one like that 

It does seem to be getting the fictious/mystery writing style and produces some fantastic and intriguing completions. Also, this is the smallest model ```ada``` of the GPT-3 family.  
