# Text Analysis using Python

## Objective
The goal of this project is to perform Text Analysis using Python to extract insights from a dataset of articles. Techniques such as word cloud generation, sentiment analysis, named entity recognition (NER), and topic modeling are applied to analyze the text data.

## Dataset used
The dataset used for this project contains articles with corresponding titles. It has been obtained from [Statso](https://statso.io/text-analysis-case-study/). Each article in the dataset is processed to extract meaningful information.

## Analysis Technique

Text Analysis involves multiple steps to derive useful insights from unstructured data. Below is an extensive explanation of the analysis techniques used:

1. **Text Preprocessing:**  
   The first step involves gathering, cleaning, and preprocessing the text data. The articles are cleaned to remove noise, irrelevant characters, and common stopwords.

2. **Word Cloud Generation:**  
   Word clouds are graphical representations of text where the size of each word indicates its frequency. In this project, the titles of the articles are combined into one string, and a word cloud is generated to visualize the most frequent words in the titles. This helps in quickly identifying the most prominent terms used in the dataset.

3. **Sentiment Analysis:**  
   Sentiment analysis involves assessing the emotional tone of the articles. Using the **`TextBlob library`**, each article’s sentiment polarity is calculated to determine whether the text expresses a positive, negative, or neutral sentiment. This provides insights into the overall mood or tone conveyed in the dataset.

4. **Named Entity Recognition (NER):**  
   NER is a technique used to identify and classify entities in the text, such as organizations, people, locations, etc. Using the **`spaCy library`**, the articles are analyzed to extract and categorize these entities. The results show the most frequently mentioned entities, providing insight into the key subjects discussed in the articles.

5. **Topic Modelling:**  
   Topic modeling is performed using **`Latent Dirichlet Allocation (LDA)`** to uncover latent topics within the articles. The text is first vectorized, converting it into a numerical format that can be understood by the LDA algorithm. The model identifies the main topics discussed across the dataset, and each article is assigned its dominant topic, helping to understand the major themes in the data.

## Result
The project successfully implements various text analysis techniques:
- A **word cloud** reveals the most frequent terms in article titles.
- **Sentiment analysis** shows the distribution of positive, negative, and neutral sentiments across articles.
- **Named entity recognition** highlights the most common entities (e.g., organizations, locations) present in the text.
- **Topic modeling** uncovers the prevalent topics discussed within the articles and visualizes their distribution.

The project demonstrates how text data can be analyzed effectively using Python, providing useful insights for data-driven decision-making. For a detailed view of the analysis and visualizations, you can access the [Jupyter Notebook here](https://nbviewer.org/gist/rijul007/7027a58fac35e282dec09f7320946fd9).