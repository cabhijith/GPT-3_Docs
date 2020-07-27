# Fine-Tuning 
Fine-tuning is training the pre-trained GPT-3 on your data. 
## Data 
The data used to fine-tune GPT-3 can either be a text (```.txt```) file or in [JSON Lines](http://jsonlines.org/)(```.jsonl```) format. JSONL is highly recommended over a 
text file. 

Saving a file in JSONL is similar to that in standard ```.json``` format. An example of saving a JSONL file: 
```python 
import jsonlines 

sample = {"data" : "GPT-3 is really good", 
         "data:" : "GPT-3 is the successor of GPT-2"
         }
with jsonlines.open( 'data/sample.jsonl' , 'w') as writer:
        writer.write_all(sample
```

The format for the data is ```{data : "<DATA>"}```, where ```<DATA>``` is: 

```foo```, a string 

```"text_list" : ["foo", "bar"]}```, a list 

```"text_list" : ["foo", "bar"], "loss_weights" : [0.1, 1]}```, a list with corresponding weights for each of the element of list. 

### Sample Dataset
We are using a dummy dataset of jokes from reddit. Here is how it looks: 
```python 
{"data": {"text_list": ["<joke>", "What do you call an Autobot who works in an overpriced makeup store at the mall ? Ulta Magnus!"], "loss_weights": [0, 1]}}
{"data": {"text_list": ["<joke>", "Why did the bicycle fall over? Because it was two-tired"], "loss_weights": [0, 1]}}
...
```
Here, the prompt ```<joke>``` is given a weight of 0 while the joke itself is given a weight of 1. This is telling the model to give all the focus to the joke and not the prompt. 

Another example: 
```python 
{"data": {"text_list": ["What is the capital of India", "New Delhi"], "loss_weights": [0.1, 0.9]}}
{"data": {"text_list": ["What is GPT-3?", "GPT-3 is a test of general knowledge."], "loss_weights": [0.1, 0.9]}}
```
Here the question is given little importance, while the answer is given much more. Alternatively, if you want to generate questions given an answer, flip the weights.

## Fine-Tuning 
Before continuing, have to have an API key and an engine sanctioned to you. Click [here](https://forms.gle/KjuDoMk21YusDUNM6) to get access to a training engine. Set your API key as environmental variable like this ```env OPENAI_API_KEY=sk-mxxxxxxxxxxxxxxxxxxxxxxxxx1```

Fine-tuning is done with the ```openai-finetune``` package. 
```python
pip install openai-finetune 
```
Basic example: 
```
openai-ft -t reddit-cleanjokes-train.jsonl --val reddit-cleanjokes-test.jsonl -e <your_engine_name> -m ada --completions-every 10 --num-completions 1 --num-epochs 5
```
```-t``` - Path to training file 

```-val``` - Path to validation file 

```-e``` - Engine name. For example ```ada-abhijith-chandran-ft-c5```

```-m``` - Specify the model. (```Ada, Babbage, Curie, DaVinci```)

```-num-epochs``` - Number of epochs to fine-tune for. 

```-completions-every``` - Every number of steps a completion has to be generated.

```-num-completions``` - Number of completions to be generated. 

## Other Parameters
```--v``` - For increased verbosity. 

```--snapshots-every 15``` - Number of steps a snapshot has to be taken. A snapshot is basically capturing the weights of the model at that time (step) and saving it forever. By taking snapshots you can further finetune them and/or use it for inference. You can find the snapshot ids in the logs outputted. Here's how it will look: ```Created snapshot ada:2020-05-03.snap-09HczpZn5OzIX70imZYslzMI```

## Inference 
Once you have fine-tuned one of the models on your data, you can get predictions from it! Here's how to get inference in python: 
```python 
engine = 'ada-louise-thomas-ft-c2'
model = 'ada:2020-05-03.snap-O4HcZMZn5OzZX70imZPalzSI #ID of the snapshot you want to get inference from 
prompt = """Standing in the doorway, she stared at the empty hallways. Suddenly"""

a = openai.Completion.create(engine=engine, model=model, prompt=prompt)
print(a['choices'][0]['text'])
```
## Terminologies/Concepts
### Tokens 
Tokens can be roughly thought of as words. Before the text input is passed on, it is tokenized where the sentences are split into words. You can read more about it [here](https://towardsdatascience.com/byte-pair-encoding-the-dark-horse-of-modern-nlp-eb36c7df4f10)

### Epoch 
An epoch is the number of times the model passes over your data. So if ```num-epochs``` is set to 5, the model in total will go 5 times over each training example. 

Different number of epochs will give different results. For example if you have little data (100,000 sentences) and train it for 100 epochs, you will lose almost 
everything the pre-trained model has learned and it will memorize your input instead of learning from it. This is called overfitting. For the GPT-3 family a 1-5 epochs are enough for most use cases unless you have a lot of training data. 

If possible, playing around with the number of epochs.

### Batch Size
Since almost all datasets are too large to be fit in a single go, it is split into batches. Batch size is how big each part is. 

Let's assume that our dataset has 16384 tokens. We set the number of tokens to be 2048 (using ```--max-tokens```). Now, if we select a batch size of 1 (2048 tokens in one go), a single epoch
will have 8 iterations. If we select a batch size of 2 we will have 4 iterations because each batch will then have 4096 tokens. Batch size will determine how fast 
your training is completed. You can increase your batch size to increase speed, but there is a limit to how big a single batch can be (depending on the engine). 

### Learning Rate
Learning rate determines how fast the model changes/adapts. If you have a small dataset, it is better to try out smaller learning rates using the ```-s``` argument. 
For example, to decrease the learning rate by 10x , use ```--s 0.1```

