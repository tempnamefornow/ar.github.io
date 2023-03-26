# Useful IDP in-house built tools


Here is a list of resuable assets that can be shared and utilized across projects and teams. These tools can be machine learning libraries, automation tools, preprocessing utilities etc.

| Name | Link | Domain | Description | Documentation |
|------|----|----------|---------| -------- |
| **idp-retain** |  [Link](https://github.com/uhg-internal/idp-retain) | Machine Learning | A standardized PyPi installable package for RETAIN preprocessing and modelling, that can be used for any project. Setting up the data preprocessing notebooks, modelling notebooks and peech library across all projects using RETAIN will introduce time wasting and overhead.vz | Documentation available on the repository README|
| **idp-geo-mapper** | [Link](https://github.com/uhg-internal/idp-geo-mapper) | Data Preprocess, time series | The IDP Geo Mapper library is a tool developed for python to convert datasets from one geo-location to another. Lets say for example you have a time series dataset that is on a county (FIPS) level, but you want to convert it to a state level. This will aggregate all the relevant data respective to each county within each state. This isn't just restricted to county to state mappings.| Read the docs [**here**](./idp_geo_map.md)|
