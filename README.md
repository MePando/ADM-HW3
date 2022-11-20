# Third Homework for Algorithmic Methods of Data Mining

## Team

- Ilaria Petrucci 1732987.
- Nicola Grieco 2081607.
- Mario Edoardo Pandolfo 1835189.

## Repository

This git repository contains our solution for the homework 3.

There are three files:

- `main.ipynb`:  a Jupyter notebook that contains all the answers to your research and theoretical questions. Clicking [here](https://nbviewer.org/github/MePando/ADM-HW3/blob/main/main.ipynb) you can find the nbviewer version of the notebook.

- `CommandLine.sh`: a bash shell script file contains the prepared script to answer to the command line question.

- `search_engines.py`: A python file containing all the classes that we have defined to create the two search engines.

  - `Vocabulary()`: The vocabulary of a `pd.DataFrame`.
  - `Index()`: The inverted index.
  - `SearchEngine()`: Our first simple search engine based on tf-idf & cosine similarity.
  - `OurSearchEngine()`: The more complex search engine.

  An example usage of these classes can be found in the `main.ipynb` notebook.

  **Note**: This python file contains only the modular classes, so there is no scraping or preprocessing section, this is because these parts depend on the data structure, while the classes contained don't: they require the data to be preprocessed (As usually an example can be found in the `main.ipynb` notebook).
