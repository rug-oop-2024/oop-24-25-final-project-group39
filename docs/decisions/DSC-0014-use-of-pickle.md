# DSC-0014: Usage pickle
# Date: 8-11-2024
# Decision: Usage of pickle in pipeline saving and loading the pipeline
# Status: Accepted
# Motivation: With Pickle, it is easy to save data to bytes and load them for our pipeline
# Reason: Needed a way to save pipeline data into bytes and load them
# Limitations: It can get compatibility issues when a saved pipeline (pickle) is loaded to a different Python version
# Alternatives: JSON, YAML