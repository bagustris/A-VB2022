
# ACII A-VB 2022 Workshop & Competition

Forked from https://github.com/HumeAI/competitions/tree/main/A-VB2022.

Code for the four tasks of A-VB 2022 competition (Team EP-ITS); our contribution is listed in `feature_based` directory. 

Full details and results can be found in the latest release of the [A-VB Paper](http://www.apsipa.org/proceedings/2022/APSIPA%202022/ThAM1-3/1570839332.pdf).

**The High-Dimensional Emotion Task (A-VB High)**

The A-VB High track explores a high-dimensional emotion space for understanding vocal bursts. Participants will be challenged with predicting the intensity of 10 emotions (Awe, Excitement, Amusement, Awkwardness, Fear, Horror, Distress, Triumph, Sadness, and Surprise) associated with each vocal burst as a multi-output regression task. Participants will report the average Concordance Correlation Coefficient (CCC) across all 10 emotions. The score for this task is set by our end-to-end approach as **0.6478 CCC**.


**The Two-Dimensional Emotion Task (A-VB Two)** 

In the A-VB Two track, we investigate a low-dimensional emotion space that is based on the circumplex model of affect. Participants will predict values of arousal and valence (on a scale from 1=unpleasant/subdued, 5=neutral, 9=pleasant/stimulated) as a regression task. Participants will report the average Concordance Correlation Coefficient (CCC), as well as the Pearson correlation coefficient, across the two dimensions. The score for this task is set by our end-to-end approach as **0.6142 CCC**.

**The Cross-Cultural Emotion Task (A-VB Culture)** 

In the A-VB Culture track, participants will be challenged with predicting the intensity of 10 emotions associated with each vocal burst as a multi-output regression task, using a model or multiple models that generate predictions specific to each of the four cultures (the U.S., China, Venezuela, or South Africa). Specifically, annotations of each vocal burst will consist of culture-specific ground truth, meaning that the ground truth for each sample will be the average of annotations solely from the country of origin of the sample. Participants will report the average Concordance Correlation Coefficient (CCC), as well as the Pearson correlation coefficient, across all 10 emotions. The score for this task is set by team is **0.4962 CCC**.

**The Expressive Burst-Type Task (A-VB Type)**

In the A-VB Type task, participants will be challenged with classifying the type of expressive vocal burst from 8 classes (Gasp, Laugh, Cry, Scream, Grunt, Groan, Pant, Other).  The score for this task is set by our team is **0.4791 UAR**.


## Citation
> B. T. Atmaja and A. Sasou, “Leveraging Pre-Trained Acoustic Feature Extractor For Affective Vocal Bursts Tasks,” in 2022 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), Nov. 2022, pp. 1412–1417, doi: 10.23919/APSIPAASC55919.2022.9980083.
