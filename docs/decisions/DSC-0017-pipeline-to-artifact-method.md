# DSC-0017: Use to-artifact method
# Date: 8-11-2024
# Decision: Extra pipeline method 'to_artifact'
# Status: Accepted
# Motivation: Using this method it is easy to save the pipeline into an artifact
# Reason: We needed to save the pipelines into the AutoML registry as an artifact
# Limitations: -
# Alternatives: Doing this everytime in the modelling file instead of in pipeline.py