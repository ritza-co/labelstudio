# Evaluating NER parsers on podcast transcripts with spaCy and Label Studio

## NER and spaCY 
*This is a shorter / condensed section as we will assume that the reader is at least partially familiar with this*.
* Quick explanation of what NER is with example
* Downloading the 'This American Life' podcast dataset
* Showing how to do NER with spaCy - show how `Easter` is often mistagged
* Show how to use different spaCy models - evaluate the small model against the large one
* But even the large one isn't that accurate in specific cases
* Therefore, label the data manually

## Manually labeling data with Label Studio
* Download and install Label Studio locally
* Label all sentences containing `Easter`
* Re-evaluate the 'large' SpaCy parser and calculate accuracy

## Where next?
* Mention that the reader can now retrain the spaCy parsers with the new data, but don't demo how to do this in this article
