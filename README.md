[👉 Link to Presententation slides](https://docs.google.com/presentation/d/1BpYxpaRu_uJYfXFw8bJLEIOOrPBZ1Kg87WyORKB3Wa8/edit?usp=sharing)

## Design Draft

#### Question/need: 
As a partner in a marriage or un-married remonatic relationship, can I identify factors that are most likely to lead to divorce, or dissolution of the relationship? Using these factors, can I predict whether a particular relationship will end in dissolution within 5 years?

#### Description of my sample data:
I processed data from a longitudinal survey of Americans in relationships performed by a group at Stanford:
> Rosenfeld, Michael J., Reuben J. Thomas, and Maja Falcon. 2018. [How Couples Meet and Stay Together, Waves 1, 2, and 3: Public version 3.04, plus wave 4 supplement version 1.02 and wave 5 supplement version 1.0 and wave 6 supplement ver 1.0](https://data.stanford.edu/hcmst) [Computer files]. Stanford, CA: Stanford University Libraries.

#### Techniques applied
This is an exploration of binary classfication and feature engineering:
 - decision trees, random forests, grid search, confusion matrices 
 - I tried out an auto-ML tool called auto-viML, with mixed results, after seeing a presentation from it's developer, (a GCP PM)
- some exploration of getting a model deployed using AWS Lambda layers and S3 (this ended up being more complicated than I had time for, and perhaps unecessary in the long run given the contraints)

#### Future Work
- A web app to allow people to input aspects of their relationship and generate a prediction.
- finalize hosting through Lambda to close the loop on the PoC
- try out some additional modeling techniques such as XGBoost


