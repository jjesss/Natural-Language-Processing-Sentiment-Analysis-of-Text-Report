% Sentiment Analysis from Text

% This project studies the dataset 'text_emotion_data_filtered.csv', and
% selects the best classification model for automatically determining the
% sentiment displayed by the tweet

% 3.1 Dataset
% The csv file has two columns, first one sit the "sentiment" column which
% lists one of four sentiments ("relief", "happiness", "surprise",
% "enthusiasm") that has been assigned to the corresponding tweet and the
% second column is the "Content" which contains the text of the tweet


% 3.2 Data Preparation
    % import csv file into MATKAB as a table
        dataTab = readtable('text_emotion_data_filtered.csv');
    % Build a Bag of Words containing all of the tokenised tweets
        documents = tokenizedDocument(dataTab.Content);
        bag = bagOfWords(documents);
    % Remove stop words
        stopWords = [stopWords "," "?" ":" "." ";" "'", "!", "`"];
        newBag = removeWords(bag,stopWords);
    % Remove any words with fewer than 100 occurrences in the bag
        refinedBag = removeInfrequentWords(newBag, 99);
        % word frequency tables to check if the right words removed:
        freq1 = topkwords(newBag, 13949);
        freq2 = topkwords(refinedBag, 77);
    % Build Term Frequency-Inverse Document Frequency Matrix
        % full() makes it not sparse
        tfidfMatrix  = full(tfidf(refinedBag));
    % Build label vector from column of sentiments
        labelsNo = categorical(dataTab.sentiment);

% 3.3 Features and Labels
    % Create a feature matrix for training of the first 6432 rows 
    % of the tf-idf matrix and all columns
    % and corresponding label vector
    trfeatures = tfidfMatrix(1:6432, :);
    trlabels = labelsNo(1:6432, :);
    % Feature matrix and label vector for testing
    tefeatures = tfidfMatrix(6433:end, :);
    telabels = labelsNo(6433:end, :);

% 4 Model Training and Evaluation
       
% 4.1 Model Training
    % the 3 classification algorithms I will be using to train and
    % comparing is KNN, disciminant analysis, decision tree
    
    % k-nearest neighbour model
    knn_model = fitcknn(trfeatures, trlabels);
    knn_predictions = predict(knn_model, tefeatures);

    % discrminant analysis model
    discr_model = fitcdiscr(trfeatures,trlabels);
    discr_predictions = predict(discr_model,tefeatures);

    % decision tree model 
    dtree_model = fitctree(trfeatures,trlabels);
    dtree_predictions = predict(dtree_model,tefeatures);
        
% 4.2 Evaluation
% compare test labels in dataset with predictions of the models:
    % accuracy = number of correct predictions/total number of labels
    labelsNo = size(telabels,1);

    % knnmodel model accuracy
    knn_correctPredictions = sum(telabels == knn_predictions);
    knn_accuracy = knn_correctPredictions / labelsNo

    % discriminant analysis model accuracy
    discr_correctPredictions = sum(telabels == discr_predictions);
    discr_accuracy = discr_correctPredictions / labelsNo

    % decision tree model accuracy
    dtree_correctPredictions = sum(telabels == dtree_predictions);
    dtree_accuracy = dtree_correctPredictions / labelsNo
    
% analysing using confusion matrix
    % knnmodel confusion matrix
    figure(1)
    knn_CM = confusionchart(telabels,knn_predictions);
    title(sprintf('KNN Model Accuracy = %.2f', knn_accuracy));

    % discriminant analysis model confusion matrix
    figure(2)
    discr_CM = confusionchart(telabels,discr_predictions);
    title(sprintf('Discriminant Analysis Model Accuracy = %.2f', discr_accuracy))

    % decision tree model confusion matrix
    figure(3)
    dtree_CM = confusionchart(telabels,dtree_predictions);
    title(sprintf('Decision Tree Model Accuracy = %.2f', dtree_accuracy))
