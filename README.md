# Climate Data Project (CDP) Analysis

## Overview
This project aims to analyze climate-related data collected through the Climate Data Project (CDP). The analysis involves exploring, cleaning, preprocessing, and analyzing textual responses from organizations regarding their strategies, financial planning, and the influence of climate-related risks and opportunities. The analysis includes sentiment scoring, clustering, dimensionality reduction, and visualization techniques to derive insights from the data.

## Project Structure
The project comprises several key components:

1. **Exploration and Cleaning**: Initial data exploration and cleaning involve loading the dataset, selecting relevant columns, handling missing values, and translating non-English responses to English using the Google Translate API.

2. **Preprocessing**: Textual data preprocessing involves removing noise, tokenization, lowercasing, removing stop words, and stemming to prepare the text for further analysis.

3. **Feature Extraction using TF-IDF**: Term Frequency-Inverse Document Frequency (TF-IDF) is utilized to extract features from the preprocessed text data, which quantifies the importance of each word in the text corpus.

4. **Dimensionality Reduction using SVD**: Singular Value Decomposition (SVD) is applied to reduce the dimensionality of the TF-IDF features while retaining essential information.

5. **Finding Ideal Number of Clusters**: The Elbow Method is employed to determine the optimal number of clusters for K-Means clustering.

6. **K-Means Clustering**: K-Means clustering is performed on the TF-IDF features to group similar responses into clusters, enabling further analysis.

7. **Sentiment Scoring**: TextBlob library is used to calculate the polarity scores for each response, determining the sentiment (positive, negative, or neutral) of each cluster.

8. **Saving to Excel**: The analyzed data, including cluster labels and sentiment scores, is saved to an Excel file for further reference.

9. **Word Cloud Analysis**: Word clouds are generated for each cluster to visualize the most frequent words in the responses, providing additional insights into the content of each cluster.

## Libraries Used
- pandas
- numpy
- nltk
- langdetect
- deep_translator
- tqdm
- scikit-learn
- yellowbrick
- TextBlob
- wordcloud
- matplotlib
- seaborn

## Usage
1. Open the `CDP.ipynb` notebook in a Jupyter environment or any compatible IDE.
2. Ensure that the necessary libraries are installed (installation commands provided in the notebook if required).
3. Run each cell sequentially to perform data analysis steps.
4. Review the generated visualizations, sentiment scores, and cluster analysis results.
5. Access the final output file `CDP_final.xlsx` for detailed insights and further analysis.

## References
- [Google Translate API Documentation](https://cloud.google.com/translate/docs)
- [TextBlob Documentation](https://textblob.readthedocs.io/en/dev/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [WordCloud Documentation](https://amueller.github.io/word_cloud/)
- [Yellowbrick Documentation](https://www.scikit-yb.org/en/latest/index.html)

## Acknowledgments
Special thanks to the CDP for providing the dataset for analysis and to all contributors to the open-source libraries used in this project.
