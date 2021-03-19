# Evaluating Named Entity Recognition parsers with spaCy and Label Studio

If you're analyzing a large amount of text, it's often useful to extract **named entities** from this - identifying people, places, dates and other entities.

While an off-the-shelf Named Entity Recognition (NER) tagger is sometimes adequate, for real-world settings you'll often need to 

1) evaluate how good the off-the-shelf solution is on your specific dataset
2) fine tune the NER tagger to your needs.

In this tutorial, we're going to focus on the evalation step. We're going to use a dataset built from transcriptions of the podcast *This American Life* and show how the standard language models that come with the NLP Python library spaCy fall short. We'll do this by comparing spaCy's predictions to a manually annotated gold standard, which we create using Label Studio.

To follow along, you should be comfortable with basic NLP and machine learning terminology (evaluation, parsing, NLP, entities, tokens) and have a local Python environment set up (be comfortable installing packages with pip or poetry and writing basic Python code).

Specifically, we will

* Download a dataset from data.world to use in our examples
* Parse this dataset with spaCy and evaluate spaCy's "small" language model against its "large" one, focusing on the token `Easter` that is often mistagged
* Manually add correct NER tags to our dataset using Label Studio
* Evaluate the larger spaCy model, against this new gold standard

## Downloading the dataset and using spaCy

As an example, let’s download the *This American Life* dataset, which is a transcript of every episode since it began in November 1995! You can download this dataset [here at data.world](https://data.world/cjewell/this-american-life-transcripts). We will be using only the `lines_clean.csv` file from here on, and specificaly the `line_text` column which contains the cleaned text data.

You can see an excerpt of the file below.

<img width="1137" alt="The columns of our dataset" src="https://user-images.githubusercontent.com/2641205/111653437-34b00980-8808-11eb-9514-eeabdd9556a7.png">

### Installing spaCy and loading our data

You'll need [spaCy](https://spacy.io) and [pandas](https://pandas.pydata.org) to continue, so install these now if you haven't already. SpaCy is a popular Python package for NLP that comes already equipped with NER, while pandas is a data frame library that we'll use to read and preprocess our CSV data

If you use Jupyter Notebook, you can follow along by creating a new notebook and recreating the code samples there. If not, create a file called `ner-evaluation.py` and add each section of code there. 

SpaCy comes with several options for pretrained models in different langauges. We'll be using `en_core_web_sm` and `en_core_web_lg` which are small and large models respectively trained on English text from the internet. In general, we would expect the small model to be more efficient but less accurate and the large model to be the opposite.

Download both models by running the following commands in your shell or command prompt:

```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

Now add the following code to your Python file:

```python 
import spacy
import pandas as pd

nlp_sm = spacy.load("en_core_web_sm")
nlp_lg = spacy.load("en_core_web_lg")


df = pd.read_csv('lines_clean.csv')
df = df[df['line_text'].str.contains("Easter ", na=False)]
print(df.head())
```

This loads the spaCy models and the dataset. We use a very crude filter to extract some data containing the word `Easter`, which we will assume has already been flagged by our team as a word that NER taggers often get wrong. 

## Parsing our data and basic evaluation

Our next step is to parse each line of our dataset and see how spaCy would tag some entities. We'll just look at the first 10 texts that contain the keyword `Easter` as an example, but in a real setting you would need a larger sample to get reliable analysis. We'll also parse each text 

Save the dataframe to a variable called `texts` and iterate through each text using a for loop. We can then apply the pre-trained pipeline package to the text and save it to the new variable `doc_sm`. In the same indented block create another for loop and iterate through each token in the document, print the token's text and entity type. 

```python
texts = df['line_text'][:10]
docs = nlp_sm.pipe(texts)
for doc in docs:    
    for token in doc:
        print(token.text, token.ent_type_)
    print("----------")
```

Here's an excerpt from the printed output: 

```
And 
then 
, 
we 
had 
Easter ORG
dinner 
at 
John 
's 
house 
on 
Irving ORG
. 
```

As you can see, this instance of 'Easter' is mistagged. The `ORG` (Organisation) tag is definitely not the right entity label here! So let's see if our model does any better when using the large spaCy model with the following code (note that we are using `nlp_lg` instead of `nlp_sm`, otherwise the code is identical.) 

```python
docs = nlp_lg.pipe(texts)
for doc in docs:    
    for token in doc:
        print(token.text, token.ent_type_)
    print("----------") 
```

Here's the same output excerpt as before:

```
And 
then 
, 
we 
had 
Easter GPE
dinner 
at 
John 
's 
house 
on 
Irving ORG
. 

```
The model makes some different predictions for the entity labels but many of them are still wrong. Instead, this time it labels the same instance of 'Easter' as `GPE` (Geopolitical Entity), which unfortunately is still the incorrect label for this entity. 

## Automatic evaluation of NER for the small and large spaCy models

It's often difficult to get a true 'gold standard' dataset, so we can bootstrap and evaluation by **assuming** that the large model is correct and seeing how the small model compares to that (even though we know from the examples above that the large model is not 100% accurate).

Replace your code with the following.

```python
import pandas as pd
import spacy

KEYWORD = "Easter"

df = pd.read_csv("lines_clean.csv")
df = df[df["line_text"].str.contains(f"{KEYWORD} ", na=False)]

texts = df["line_text"]

print(df.head())
print(df.shape)

nlp_sm = spacy.load("en_core_web_sm")
nlp_lg = spacy.load("en_core_web_lg")

docs_sm = list(nlp_sm.pipe(texts))
docs_lg = list(nlp_lg.pipe(texts))

total_tokens = 0
agreed_tokens = 0
total_matches = 0
agreed_matches = 0

# look at each text
for i in range(len(texts)):
    # print(docs_sm[i])
    doc_sm = docs_sm[i]
    doc_lg = docs_lg[i]

    # look at each token in that text
    for i in range(len(doc_sm)):
        total_tokens += 1
        if doc_sm[i].text == f"{KEYWORD}" and doc_lg[i].text == f"{KEYWORD}":
            total_matches += 1
            print(doc_sm[i - 5 : i + 5])
            print(
                f"spacy_sm: {doc_sm[i].ent_type_} {doc_sm[i].text} spacy_lg: {doc_lg[i].ent_type_} {doc_lg[i].text} "
            )
            print("---")

            if doc_sm[i].ent_type == doc_lg[i].ent_type:
                agreed_matches += 1

        if doc_sm[i].ent_type is not None:
            if doc_sm[i].ent_type == doc_lg[i].ent_type:
                agreed_tokens += 1

print(
    f"""
Total tokens processed: {total_tokens}
Small and large model agreed on {agreed_tokens} ({(agreed_tokens/total_tokens):.2f}%)
Keywords ({KEYWORD}) processed: {total_matches}
Small and large model agreed on {agreed_matches} ({(agreed_matches/total_matches):.2f}%)
    
"""
)  
```

This is quite a large chunk of code, but most of it is boilerplate to set up spaCy and some basic variables to evaluate the results. We extract all of the 
documents which match our keyword, and then loop through each token in each doc, comparing the predictions of the different models and keeping track of how 
often they agree.

For each match, we also print out 10 words of context so we can do some basic sanity checking and qualitative evaluation too.

Here's a sample of the output you should see.

```
And then, we had Easter dinner at John's
spacy_sm: ORG Easter spacy_lg: GPE Easter 
---
When? 11:30 PM, Easter eve. Where?
spacy_sm: PERSON Easter spacy_lg: DATE Easter 
---
Super Sport, night before Easter, Route 11.
spacy_sm: GPE Easter spacy_lg: DATE Easter 
---
Our viewers saw art last Easter with a two-
spacy_sm:  Easter spacy_lg:  Easter 
---
. We went back for Easter and then Thanksgiving and
spacy_sm: PERSON Easter spacy_lg:  Easter 

<...>

Total tokens processed: 3286
Small and large model agreed on 3182 (0.97%)
Keywords (Easter) processed: 39
Small and large model agreed on 6 (0.15%)
```

Despite spaCy being incredibly useful for a range of NLP tasks, its predictions for our entities are not always correct. If the "Easter" issue is important for our project, we're going to need to manually label the entities ourselves.

## Labelling Named Entities in Label Studio

[Label Studio](https://labelstud.io) is an open source data labeling tool. You can install it with `pip` or in the same way you usually install Python libraries. Run the following in your terminal if you haven't already.

```
pip install -U label-studio
```

When you start Label Studio for the first time, it launches from a project directory that Label Studio creates. To get started type the following into your terminal:

`label-studio init ner-tagging`

And now you can start up the interface by running 

`label-studio start ./ner-tagging` 

This will automatically launch a localhost webpage in your browser and direct you to the Label Studio website. From here you can import your data (`lines_clean.csv` in our case) and start labelling. To get started we also need to choose our type of labelling project (Named Entity Recongition) and set our label configurations.

![Screenshot 2021-03-17 at 14 19 15](https://user-images.githubusercontent.com/66478571/111475477-686a3100-872d-11eb-9815-70221d84da18.png)

```
<View>
  <Labels name="ner" toName="text">
    <Label value="Person"></Label>
    <Label value="Organization"></Label>
    <Label value="Fact"></Label>
    <Label value="Money"></Label>
    <Label value="Date"></Label>
    <Label value="Time"></Label>
    <Label value="Ordinal"></Label>
    <Label value="Percent"></Label>
    <Label value="Product"></Label>
    <Label value="Language"></Label>
    <Label value="Location"></Label>
  </Labels>
  <Text name="text" value="$text"></Text>
</View>
```
Change 'text' in `$text` to line_text and hit 'save'.

Next we filter the lines using the Label Studio user interface. Again, we are interested in the line_text column and those lines that contain “Easter”. You can select “Filters” as follows: 

![Screenshot 2021-03-15 at 15 26 54](https://user-images.githubusercontent.com/66478571/111380238-5d1ef300-86a4-11eb-82c5-63913653a774.png)

Label Studio comes equipped with the entity labels: Person, Organization, Fact, Money, Date, Time, Ordinal, Percent, Product, Language, and Location. To label a word, first select the label you want to assign and then highlight the word/s in the text. Here I labelled “Easter Sunday” with the Time tag: 

![Screenshot 2021-03-15 at 15 28 15](https://user-images.githubusercontent.com/66478571/111380202-54c6b800-86a4-11eb-979a-7ba177878cea.png)


Once you've finished labelling you can export the dataset that you labeled, including the completed annotations, or just export the annotations from Label Studio.

Label Studio stores your annotations in a raw `json` format in the `my_project_name/completions directory`, or whichever cloud or database storage you specify as your target storage, with one file per labeled task named as task_id.json. You can also choose another file type and it will reformat the data for you, in this case I chose to export the file as a `csv.

The csv format for the current task should look something like this: 

```csv
act_name	episode_id	line_text	speaker	speaker_class	timestamp	id	label			
Act One	1	So I entered Jerusalem on Easter with a simple expectation that I was going to photograph another religious ceremony, another religious festival. And then, for various reasons, I got locked out of my hostel room. They had a curfew. And I didn't make it back in time. And I was in quite a fix because I was a stranger in this very strange town. When it happened, I didn't have enough money to stay elsewhere, nor did I even have knowledge of where to go.	Kevin Kelly	host	00:08:06	36	[{"end": 32, "labels": ["Time"], "start": 26, "text": "Easter"}]			

```
Now we know our own entity labelling in Label Studio is the gold standard so let's compare it with the predicted labels of the spaCy large model. Open the Label Studio results file with `pd.read_csv()` and save it to the variable `result`, then since we're only interested in the column `label` we can then get rid of the rest and just save that column to the variable `labels`. 
```python
result = pd.read_csv('result.csv')
labels = result.label
```
Next initate an empty list. The `labels` column is formatted like a dictionary in Python so we're going to iterate throught it and append each value to our empty list. 
```python
label_studio_list = []
for key, value in labels.items():
      label_studio_list.append(value)     
``` 
When we print `label_studio_list` we can see there's still a lot of unnecessary information in it:

```
print(label_studio_list)
['[{"end": 32, "labels": ["Time"], "start": 26, "text": "Easter"}]', '[{"end": 95, "labels": ["Time"], "start": 89, "text": "Easter"}]', '[{"end": 23, "labels": ["Time"], "start": 17, "text": "Easter"}]', '[{"end": 30, "labels": ["Person"], "start": 18, "text": "Easter Bunny"}]', '[{"end": 159, "labels": ["Person"], "start": 147, "text": "Easter bunny"}]', '[{"end": 45, "labels": ["Person"], "start": 33, "text": "Easter Bunny"}]', '[{"end": 236, "labels": ["Time"], "start": 230, "text": "Easter"}]', '[{"end": 49, "labels": ["Person"], "start": 37, "text": "Easter Bunny"}]', '[{"end": 32, "labels": ["Time"], "start": 26, "text": "Easter"}]', '[{"end": 58, "labels": ["Person"], "start": 46, "text": "Easter Bunny"}]', '[{"end": 233, "labels": ["Time"], "start": 227, "text": "Easter"}]', '[{"end": 14, "labels": ["Time"], "start": 8, "text": "Easter"}]', '[{"end": 499, "labels": ["Time"], "start": 493, "text": "Easter"}]', '[{"end": 203, "labels": ["Time"], "start": 197, "text": "Easter"}]', '[{"end": 349, "labels": ["Time"], "start": 343, "text": "Easter"}]', '[{"end": 172, "labels": ["Product"], "start": 162, "text": "Easter egg"}]', '[{"end": 303, "labels": ["Product"], "start": 290, "text": "Easter basket"}]', '[{"end": 189, "labels": ["Product"], "start": 177, "text": "Easter dress"}]', '[{"end": 27, "labels": ["Time"], "start": 14, "text": "Easter Sunday"}]', '[{"end": 628, "labels": ["Person"], "start": 616, "text": "Easter bunny"}, {"end": 92, "labels": ["Person"], "start": 80, "text": "Easter bunny"}]', '[{"end": 341, "labels": ["Time"], "start": 335, "text": "Easter"}]', '[{"end": 180, "labels": ["Time"], "start": 174, "text": "Easter"}]', '[{"end": 34, "labels": ["Time"], "start": 20, "text": "Easter weekend"}]', '[{"end": 13, "labels": ["Time"], "start": 7, "text": "Easter"}]', '[{"end": 140, "labels": ["Product"], "start": 129, "text": "Easter eggs"}]', '[{"end": 33, "labels": ["Time"], "start": 27, "text": "Easter"}]', '[{"end": 145, "labels": ["Product"], "start": 134, "text": "Easter eggs"}]', '[{"end": 818, "labels": ["Person"], "start": 806, "text": "Easter Bunny"}]', '[{"end": 355, "labels": ["Time"], "start": 341, "text": "Easter morning"}]', '[{"end": 54, "labels": ["Time"], "start": 48, "text": "Easter"}]', '[{"end": 172, "labels": ["Product"], "start": 161, "text": "Easter eggs"}]', '[{"end": 202, "labels": ["Person"], "start": 190, "text": "Easter bunny"}]', '[{"end": 265, "labels": ["Time"], "start": 252, "text": "Easter Sunday"}]', '[{"end": 172, "labels": ["Product"], "start": 162, "text": "Easter egg"}]', '[{"end": 316, "labels": ["Time"], "start": 310, "text": "Easter"}]']
```

We can use a list comprehension to split the list by ',' and isolate the `"labels":` and the proceeding entity label, which is at index 1 in each list. 

```python
label_studio_list = [i.split(',')[1] for i in label_studio_list]
```
```
>>> [' "labels": ["Time"]', ' "labels": ["Time"]', ' "labels": ["Time"]', ' "labels": ["Person"]', ' "labels": ["Person"]', ' "labels": ["Person"]', ' "labels": ["Time"]', ' "labels": ["Person"]', ' "labels": ["Time"]', ' "labels": ["Person"]', ' "labels": ["Time"]', ' "labels": ["Time"]', ' "labels": ["Time"]', ' "labels": ["Time"]', ' "labels": ["Time"]', ' "labels": ["Product"]', ' "labels": ["Product"]', ' "labels": ["Product"]', ' "labels": ["Time"]', ' "labels": ["Person"]', ' "labels": ["Time"]', ' "labels": ["Time"]', ' "labels": ["Time"]', ' "labels": ["Time"]', ' "labels": ["Product"]', ' "labels": ["Time"]', ' "labels": ["Product"]', ' "labels": ["Person"]', ' "labels": ["Time"]', ' "labels": ["Time"]', ' "labels": ["Product"]', ' "labels": ["Person"]', ' "labels": ["Time"]', ' "labels": ["Product"]', ' "labels": ["Time"]']
```
Then we can clean the lines some more by removing `"labels":` and the square brackets. 

```python
label_studio_list = ([s.replace('"labels": ', '') for s in label_studio_list])
label_studio_list = ([s.replace('["', '') for s in label_studio_list])
label_studio_list = ([s.replace('"]', '') for s in label_studio_list])
print(label_studio_list)
```
```
>>> [' Time', ' Time', ' Time', ' Person', ' Person', ' Person', ' Time', ' Person', ' Time', ' Person', ' Time', ' Time', ' Time', ' Time', ' Time', ' Product', ' Product', ' Product', ' Time', ' Person', ' Time', ' Time', ' Time', ' Time', ' Product', ' Time', ' Product', ' Person', ' Time', ' Time', ' Product', ' Person', ' Time', ' Product', ' Time']

```
Now we have a list of only the entity labels we annotated in Label Studio. Let's compare the labels agaist those of our spaCy large model use the `zip()` function to iterate through the tuples with the corresponding elements from each of the lists, which we can the format. Finally, just join the lists together using `join()` and `print` the result. 

```python
results_comparison = "\n".join("{} {}".format(x, y) for x, y in zip(spacy_large_list, label_studio_list))
print(results_comparison)
```
```
>>>   Time
  Time
TIME  Time
  Person
  Person
  Person
TIME  Time
GPE  Person
GPE  Time
DATE  Person
DATE  Time
  Time
  Time
  Time
  Time
NORP  Product
  Product
  Product
  Time
  Person
LOC  Time
DATE  Time
  Time
  Time
  Product
NORP  Time
  Product
  Person
GPE  Time
  Time
  Product
DATE  Person
DATE  Time
GPE  Product
EVENT  Time
```
As we can see the spaCy large model rarely made the same predictions for the entity labels as the labels made in Label Studio, our gold standard. For a model to make accurate predictions it needs training data. The more relevant that training data is to the task, the more accurate the model will be at completing said task. We could now retrain the spaCy parsers with the new data from Label Studio. Using Label Stdio's simple and straightforward UI we can label data quickly and use it to improve existing training data to get more accurate ML models.
