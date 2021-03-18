# NER with Label Studio
Named Entity Recognition (NER) is the task of categorising key words in a text as specific entities. An entity can be any word or series of words that consistently refers to the same thing. Each detected entity is classified into a predetermined category, making NER a powerful tool for analysing and categorising natural language. NER can be helpful in getting a high-level overview of a text, understanding the main themes of a body of texts, or grouping texts together based on their similarity. NER is used in many fields in Artificial Intelligence, Natural Language Processing, and Machine Learning for applications ranging from customer support to academia. 

## spaCy for NER
SpaCy is a popular Python package for NLP that comes already equipped with NER as well as some other handy features for Text Classification, Tokenization and Part of Speech (POS) tagging among others. To see how it works let’s download The American Life dataset, a transcript of every line of every podcast episode since it began in November 1995! This is a great dataset to play around with since you can download the already cleaned lines from [here](https://data.world/cjewell/this-american-life-transcripts). We will be using the file: `lines_clean.csv`.

First, create a python file and import the spacy and pandas modules. SpaCy comes with a pre-trained pipeline package that identifies tokens fitting a pre-determined set of named entities. Download spaCy small and save it to the variable `nlp_sm` like so:
```python 
import spacy
import pandas as pd

spacy.cli.download("en_core_web_sm") ## spacy small
nlp_sm = spacy.load("en_core_web_sm")
```

Next you can load `lines_clean.csv` and save it to the variable `df_sm`. In this project we will see how spaCy tags (or often mistags) the entity "Easter", therefore we are only interested in the data column ‘line_text’ and only those lines that contain “Easter “, note the intentional whitespace after Easter, this will stop us capturing words like “Easterling”. 

```python
df = pd.read_csv('lines_clean.csv')
df = df[df['line_text'].str.contains("Easter ", na=False)]
```
Save the dataframe to a variable called `texts` and iterate through each text using a for loop. We can then apply the pre-trained pipeline package to the text and save it to the new variable `doc_sm`. In the same indented block create another for loop and iterate through each token in the document, print the token's text and entity type. 

```python
texts = df['line_text']
for text in texts[:10]:
    doc_sm = nlp_sm(text)
    
    for token in doc_sm:
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
As you can see, many instances of 'Easter' are mistagged, `ORG` (Organisation) is definitely not the right entity label here! So let's see if our model does any better when using spaCy large. Again, we want to download the pre-trained pipeline packaging but this time specifying `lg` rather than `sm`. 

```python
spacy.cli.download("en_core_web_lg") ## spacy large
nlp_lg= spacy.load("en_core_web_lg")
```
And go through the same steps as before by iterating through the 
```
for text in texts[:10]:
    doc_sm = nlp_lg(text)


    for token in doc_lg:
        print(token.text, token.ent_type_)
    print("----------")
 ```

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


## Comparing NER results for spaCy small and spaCy large

Let's compare the results of the predicted label entities for the spaCy small model and the spaCy large model. Inside the same for loop as before deactivate the print statements with the `#` character.

```python
for text in texts:
    print(text)
    doc_sm = nlp_sm(text)
    doc_lg = nlp_lg(text)
    for token in doc_sm:
        pass
        # print(token.text, token.ent_type_)
    for token in doc_lg:
        pass
        # print(token.text, token.ent_type_)
    print("----------")
```

Create another for loop nested within this previous for loop. Here we will iterate through the entities in  both `doc_sm` and `doc_lg`, we do this by calling `ent1` and `ent2`, and zipping the two files together: `zip(doc_sm, doc_lg)`. Since we're only concerned with the 'Easter' entities, next we create an if statement inside the for loop that prints out the entities in `doc_sm` and `doc_lg` if the entity matches the string 'Easter', `ent1.ent_type_ == ent2.ent_type_` will also tell us if the entities types are a match or not (True or False). For the second if statement delcare the variable `spacy_large_list` outside the entire for loop and create an empty list. For the third if statement declare the variable `count` outside the entire for loop again and set it to 0, this if statement will keep track of how many times the entity type for 'Easter" is the same for both models by incrementing the variable `count` by 1 each time it is. 

```python
spacy_large_list = []
count = 0
```
```python
    for ent1, ent2 in zip(doc_sm, doc_lg):
        if ent1.text == 'Easter':
            print(ent1.text, ent2.text, ent1.ent_type_, ent2.ent_type_, ent1.ent_type_ == ent2.ent_type_)
        if ent2.text == 'Easter':
            spacy_large_list.append(ent2.ent_type_)
        if ent1.text == 'Easter' and (ent1.ent_type_ == ent2.ent_type_) == True:
            count += 1

         
``` 
This will print the following text for all instances of 'Easter' in the text. Here we can clearly compare the results of the two models for each instance:

```
>>>
And then, we had Easter dinner at John's house on Irving. A big sit-down meal in daytime's unnerving. For me, anyway, I like to eat later. I took some roast lamb, a couple potaters, and carried my plate outside to the lawn, to a table he'd covered in bright pink chiffon. I'd had a few beers and a glass of chablis and thought, before eating, I might want to pee.
----------
Easter Easter ORG GPE False
```
Then make a print statement outside all for loops with the string "Total number of entity label matches for Easter: " and the `count` variable so we can see how many times both models predicted the same entity label:

```python
print("Total number of entity label matches for Easter: ", count)
```
```
>>> Total number of entity label matches for Easter:  6
```
Unfortunately 6 matches isn't a great result, and even the spaCy large model isn't accurate in specific cases. 

## NER in Label Studio

Despite spaCy being incredibly useful for a range of NLP tasks its predictions for our entities are not always correct, so it looks like we're going to need to manually label the entities ourselves. Luckily, Label Studio can help us do just that. Label Studio is an open source data labeling tool for labeling and exploring multiple types of data. Label Studio can be integrated with machine learning models to supply predictions for labels (pre-labels), or perform continuous active learning.  You can use Label Studio for a variety of labelling and classification tasks for many different data formats but again we will just be focusing on its NER capabilities. 

When you start Label Studio for the first time, it launches from a project directory that Label Studio creates, called ./my_project by default. To get started type the following into your terminal:

`label-studio init my_project`

`my_project` can be replaced by any title you like, this will be the name for your project where all the labelling activities in Label Studio will occur. Next we want to start the server, copy the following into your terminal:

`label-studio start ./my_project` 

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
