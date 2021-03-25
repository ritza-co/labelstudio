# Evaluating Named Entity Recognition parsers with spaCy and Label Studio

If you're analyzing a large amount of text, it's often useful to extract **named entities** from this – identifying people, places, dates and other entities.

While an off-the-shelf Named Entity Recognition (NER) tagger is sometimes adequate, for real-world settings you'll often need to: 

1) Evaluate how good the off-the-shelf solution is on your specific dataset.
2) Fine-tune the NER tagger to your needs.

In this tutorial, we're going to focus on the evalation step. We're going to use a dataset built from transcriptions of the podcast *This American Life* and show how the standard language models that come with the NLP Python library spaCy fall short. We'll do this by comparing spaCy's predictions to a manually annotated gold standard, which we create using Label Studio.

To follow along, you should be comfortable with basic NLP and machine learning terminology (evaluation, parsing, NLP, entities, tokens) and have a local Python environment set up (be comfortable installing packages with pip or poetry and writing basic Python code).

Specifically, we will:

* Download a dataset from data.world to use in our examples.
* Parse this dataset with spaCy and evaluate spaCy's "small" language model against its "large" one, focusing on the token `Easter` that is often mistagged.
* Manually add correct NER tags to our dataset using Label Studio.
* Evaluate the larger spaCy model, against this new gold standard.

## Downloading the dataset and using spaCy

We are going to find a large sample of text to analyze. For the purposes of this tutorial, let’s download the *This American Life* dataset, which is a transcript of every episode since it began in November 1995! 

You can download this dataset [here at data.world](https://data.world/cjewell/this-american-life-transcripts). We will be using only the `lines_clean.csv` file from here on, and specificaly the `line_text` column which contains the cleaned text data. _WHY_

You can see an excerpt of the file below.

<img width="1137" alt="The columns of our dataset" src="https://user-images.githubusercontent.com/2641205/111653437-34b00980-8808-11eb-9514-eeabdd9556a7.png">

### Installing spaCy

You'll need [spaCy](https://spacy.io) and [pandas](https://pandas.pydata.org) to continue, so install these now if you haven't already. 
* SpaCy is a popular Python package for NLP that comes already equipped with NER.
* Pandas is a data frame library that we'll use to read and preprocess our CSV data.

If you use Jupyter Notebook, you can follow along by creating a new notebook and recreating the code samples there. 
If not, create a file called `ner-evaluation.py` and add each section of code there. 

SpaCy comes with several options for pretrained models in different langauges. We'll be using `en_core_web_sm` and `en_core_web_lg` which are small and large models respectively trained on English text from the internet. In general, we would expect the small model to be more efficient but less accurate and the large model to be the opposite.

Download both models by running the following commands in your shell or command prompt:

```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```
### Loading our data

Let's loads the spaCy models and the dataset. Add the following code to your Python file:

```python 
import spacy
import pandas as pd

nlp_sm = spacy.load("en_core_web_sm")
nlp_lg = spacy.load("en_core_web_lg")


df = pd.read_csv('lines_clean.csv')
df = df[df['line_text'].str.contains("Easter ", na=False)]
print(df.head())
```

We use a very crude filter to extract some data containing the word `Easter`, which we will assume has already been flagged by our team as a word that NER taggers often get wrong. 

## Parsing our data and basic evaluation

Our next step is to parse each line of our dataset and see how spaCy would tag some entities. We will:
+ Look at the first 10 texts that contain the keyword `Easter` as an example (in a real setting, you would need a larger sample to get reliable analysis). 
+ Parse each text.
+ Save the dataframe to a variable called `texts`. 
+ Iterate through each text using a for loop. 

We can then apply the pre-trained pipeline package to the text and save it to the new variable `doc_sm`. In the same indented block, create another for loop and iterate through each token in the document. Then, print the token's text and entity type. 

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

```
label-studio init ner-tagging
```

Complete the initialisation steps as prompted.

<img width="908" alt="CleanShot 2021-03-19 at 14 33 53@2x" src="https://user-images.githubusercontent.com/2641205/111788264-2cb3a080-88c0-11eb-858b-08a8a7908b79.png">


```
label-studio start ner-tagging
```

Sign up or log in in when prompted.
<img width="1122" alt="CleanShot 2021-03-19 at 14 36 23@2x" src="https://user-images.githubusercontent.com/2641205/111788581-8ae08380-88c0-11eb-91ad-fd169e72c864.png">

And you'll be taken to your project dashboard, showing the `ner-tagging` project we just created. Click on that, and choose to import your data from a local file. Upload the `lines_clean.csv` file and specify that it is a `List of tasks`. Then press the import data button.

![CleanShot 2021-03-19 at 14 39 48@2x](https://user-images.githubusercontent.com/2641205/111789048-0b06e900-88c1-11eb-9195-a454ab3ef6a5.png)
<img width="1349" alt="CleanShot 2021-03-19 at 14 39 01@2x" src="https://user-images.githubusercontent.com/2641205/111788884-e27eef00-88c0-11eb-8fc3-3ff4338611e6.png">

You'll see all the data be imported into the labelling interface. To make it more manageable, let's add a filter for 'Easter ' again, so we can focus on that data. Click on 'Filters', select `line_text` and type `Easter ` (with the trailing space again as a crude way to avoid other words like `Eastern`.)

![CleanShot 2021-03-19 at 14 42 47@2x](https://user-images.githubusercontent.com/2641205/111789496-7fda2300-88c1-11eb-8308-ddd6156290a6.png)

Click on one of the data rows and you will be prompted to add a labelling configuration. Add the following configuration, specifying that we are interested in the data in the `line_text` column and specifying some standard NER labels as possible labels.

```xml
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
  <Text name="text" value="$line_text"></Text>
</View>
```

Hit save, and you'll be taken back to your data. Now click on a row and you can add labels manually to specific words or phrases. Use the helpful pre-defined shortcuts (`1`, `2`, etc) to quickly select the correct label before selecting the relevant portion of text.

![](https://cln.sh/9WZBul+)

Once you've finished labelling you can export the dataset that you labeled. We'll export our labels as `.csv` to match the dataset that we were working on.

![CleanShot 2021-03-19 at 14 52 09@2x](https://user-images.githubusercontent.com/2641205/111790696-c3815c80-88c2-11eb-96b1-afd57a65cf36.png)

The csv format for the current task should look something like this: 

<img width="1194" alt="CleanShot 2021-03-19 at 16 15 09@2x" src="https://user-images.githubusercontent.com/2641205/111802484-54aa0080-88ce-11eb-83dc-d56723dd7ae4.png">


Let's import this data back into our original project and compare it to spaCy. We'll just use this sample of eight manually done labels as an example, but in a real project you would likely need to do a lot more for meaningful results. Note that what the "correct" tag for Easter *should* be is pretty ambiguous, even for humans - we've used `Date` in most examples and `Person` for "Easter Bunny", but what you choose will depend on your project and needs.


## Comparing the spaCy model against our new gold standard

Rename the file you downloaded from Label Studio to `manual-easter-labels.csv` (it will have a long name with a time stamp by default). Move this file to the same directory as the data and the `ner-evaluation.py` Python script we were working on before.

Remove all the existing code in the `ner-evaluation.py` file and replace it with the following.

```python
import pandas as pd
import json

import spacy

nlp = spacy.load('en_core_web_lg')

manual_labels = pd.read_csv('manual-easter-labels.csv')
manual_labels.head()

l = manual_labels[['line_text', 'ner']]

for i, text in enumerate(manual_labels['line_text']):
    gold_labels = set(json.loads(manual_labels['ner'][i])[0]['labels'])
    
    doc = nlp(text)
    spacy_labels = {token.ent_type_ for token in doc if token.ent_type_}

    print(f"""
{text} ...
spaCy labels: {spacy_labels}
gold_labels: {gold_labels}
""")
```

This doesn't calculate any metrics like we did before. In our example, we've only labelled a few texts and only focused on the keyword `Easter`, so it's not really 'fair' to compare this to spaCy's versions, which are more complete.

However, it should produce output similar to the following excerpt

```

I slept there until early morning, when the activity started to increase, and people started coming in. And I went out and followed the crowd where it was going when they were going out to the tombs area in Jerusalem. And I went out. And there were some folding chairs set up in front of this tomb area. And as the sun was coming up on that Easter morning, I was staring at empty tombs. And for a reason that I can not comprehend, as I sat on that chair contemplating this view of the early sun morning coming into the empty tombs, all that I had been wrestling with for the past many, many years in thinking about religion sort of became resolved in my mind. And at that very moment, I believed that Jesus Christ had, indeed, risen from those tombs. ...
spaCy labels: {'GPE', 'TIME'}
gold_labels: {'Date'}
```

Showing how spaCy and gold labels differ for each text. If we invest the manual time and effort into really labelling each example by hand, we could compare the two sets of labels and see how often spaCy gets them right. 

## Where next?

You've seen how to do basic NER tagging, both automatically and manually. For a real-world use case, you would need to manually label a large amount of data specific to your project, and then [retrain spaCy's models](https://spacy.io/usage/training) based on this new data set.

We've also taken some shortcuts in the above code, such as using `Easter ` as a crude filter, which will skip examples where our keyword is followed by a punctuation mark and only calculating very crude accuracy metrics, but you should be able to adapt these samples to your specific needs.


