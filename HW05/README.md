## CS 383 Homework 5 - Logistic Regression

Name: Cecilia

userId:

Number of Late Days Using for this homework:

---

### Analysis Questions

1. So far, we have run logistic regression and Naive Bayes on different types of datasets. Why is that? Discuss the types of features and labels that we worked with in each case.

Logistic regression and Naive Bayes were applied to different datasets to suit their modeling strengths. The phoneme classification task uses speech signals for the phonemes “aa” and “ae”. Here, continuous features are derived from acoustic measurements such as frequency and amplitude, and binary labels indicate which phoneme appears. This setup is well suited for logistic regression, which effectively handles linear relationships in binary classification problems.

In contrast, the Naive Bayes classifier analyzes the Zoo dataset, classifying animals into categories such as mammals, birds, and reptiles. The features in this dataset are discrete, representing characteristics such as the presence of feathers or habitat type. The labels are multiclass, ranging from 0 to 6 for different animal classes.Naive Bayes specializes in categorical data and assumes conditional independence between features. Therefore, Logistic regression is suitable for continuous features and binary labels, whereas Naive Bayes is suitable for discrete features and multi-class labels.


2. Explain how you could transform the CSV dataset with *continuous features* to work for Naive Bayes. You
could use something similar to what we did for decision trees, but try to think about alternatives.
Maybe use something like Gaussian Naive Bayes, with the assumption of the features follow a normal distribution, allowing naive bayes to maintain the continuous nature of the data.

3. Then explain how you could transform the ARFF datasets with *discrete features* to work with logistic regression. What issues need to be overcome to apply logistic regression to the zoo dataset?

We need to encode the categorical variables into numerical values. suppose we have dog, cat and fish as animal as categorical varaible. we now can convert those into cat - 1, fish - 0, and dog - 1.5.

---

### Lab Questionnaire

(None of your answers below will affect your grade; this is to help refine homework assignments in the future)

1. Approximately, how many hours did you take to complete this homework? (provide your answer as a single integer on the line below)
5

2. How difficult did you find this homework? (1-5, with 5 being very difficult and 1 being very easy)
4

3. Describe the biggest challenge you faced on this homework:
