<img src="../../../../resources/images/developer_handbook/tools/logo.png" style="height: 300px; display: block; margin-right: auto; margin-left: auto;"> 

## Overview 👋

The <b>Optum-geo</b> library is a tool developed for python to convert datasets from one geo-location to another. Lets say for example you have a time series dataset that is on a county (FIPS) level, but you want to convert it to a state level. This will aggregate all the relevant data respective to each county within each state. 

This isn't just restricted to county to state mappings. Check the available mappings below: 

- Zip3 → FIPS
- FIPS → CBSA
- FIPS → CSA
- FIPS → State
- FIPS → Micropolitan Statistical Area
- FIPS → Metropolitan Statistical Area

???- question "Understanding USE Geo Resolutions🗺"
    <h2>County Level FIPS</h2> 

    - FIPS the lowest resolution in this library. A Federal Information Processing Series is 5 digit numeric code. The first two digits relate to US states and the three digits relate to the county. (3142 counties)
    - Example: ```FIPS 06071 relates to, 06-California and 071-San Bernardino County``` 

    ---

    <h2>Core Based Statistical Areas (CBSA)</h2>
    CBSAs represent geographic entities based on the concept of a core area with a large population nucleus, plus adjacent communities having a high degree of social and economic integration with that core. 

    - A CBSA consists of a U.S. county or counties associated with at least one urban core ( with a population of at least 10,000) along with any adjacent counties having a high degree of social and economic integration with the core as measured through commuting ties with the counties containing the core. 
    - The offical list of CBSA can be obtained from [US Census Delineation Files](https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html)
    - CBSA are further categorized as 
        - __Metropolitian statisitical area MSA__ where the urbanized core of the CBSA is 50,000 or more population (392 such areas, 1180 counites,  representing 86.1% of the US population)
        - __Micropolitan statistical areas__ where the urban cluster is of at least
    - 10,000 and less than 50,000 population (546 such areas, 661 counites, representing 8.3% of the US population) 
    -   ```Codes for Metropolitan and Micropolitan Statistical Areas and Metropolitan Divisions fall within the 10000 to 49999 range and are assigned in alphabetical order by area title. Metropolitan Divisions are 
    distinguished by a5-digit code ending in 4.``` 

    ---


    <h2>Combined Statiscal Areas (CSA)</h2>
    are adjacent Metropolitan and Micropolitan Statistical Areas, in various combinations,  that have social and economic ties as measured by commuting, but at lower levels than are found among counties within CBSAs. This can be characterized as representing larger regions that reflect broader social and economic interactions, such as wholesaling, commodity distribution, and weekend recreation activities. (175 such areas, representing 551 metro & micro CBSA)
    - ```Combined Statistical Area and Combined NECTA(New England City & Town Areas) codes are 3 digits in length. Combined Statistical Area codes fall within the 100 to 599 range. Combined NECTA codes fall within the 700 to 799 range.``` 

    _A detailed explanation of the above can be found at [Office of Management Budget](https://www.whitehouse.gov/wp-content/uploads/2020/03/Bulletin-20-01.pdf)._

    <img src="../../../../resources/images/developer_handbook/tools/geo_readme.png" style="height: 300px; display: block; margin-right: auto; margin-left: auto;"> 

    ---

    <h2>Zone Improvement Plan ZIP Codes </h2>
    are probably the most recognisable of all geo codes. Zip codes are not drawn according to to county boundaries. They are created by the the United States Postal Service. 

    - 20% of ZIP codes lie in more than 1 county. 

    ```ZIP is broken down along national area (digit 1), region or city (digit 2-3), delivery/postal office area(digts 4-5)``` 

    <img src="../../../../resources/images/developer_handbook/tools/zip_code_map.png" style="height: 400px; display: block; margin-right: auto; margin-left: auto;"> 


## Example
Here is an example of how the library will convert FIPS level data to CBSA level. The CBSA 26700 is made up of two counties (46073 and 46005). Alternatively, see [the examples notebook](https://github.com/uhg-internal/idp-geo-mapper/blob/master/notebooks/examples.ipynb) for more examples on how to use it.

**Input**

- Example of a datafame in FIPS level with 46073 and 46005 FIPS
- This data is used as input


|fips|county                       |province_state|country                                      |date      |confirmed|
|----|-----------------------------|--------------|---------------------------------------------|----------|---------|
|46073|Jerauld                      |South Dakota  |US                                           |2020-07-05|39       |
|46073|Jerauld                      |South Dakota  |US                                           |2020-07-06|39       |
|46073|Jerauld                      |South Dakota  |US                                           |2020-07-07|39       |
|46073|Jerauld                      |South Dakota  |US                                           |2020-07-08|39       |
|46073|Jerauld                      |South Dakota  |US                                           |2020-07-09|39       |
|46005|Beadle                       |South Dakota  |US                                           |2020-07-05|540      |
|46005|Beadle                       |South Dakota  |US                                           |2020-07-06|541      |
|46005|Beadle                       |South Dakota  |US                                           |2020-07-07|545      |
|46005|Beadle                       |South Dakota  |US                                           |2020-07-08|549      |
|46005|Beadle                       |South Dakota  |US                                           |2020-07-09|551      |

**Output**

- The package will output an aggregated version on a CBSA level

|confirmed|cbsa_name                    |state |date                                         |cbsa      |
|---------|-----------------------------|------|---------------------------------------------|----------|
|579      |Huron, SD                    |South Dakota|2020-07-05                                   |26700     |
|580      |Huron, SD                    |South Dakota|2020-07-06                                   |26700     |
|584      |Huron, SD                    |South Dakota|2020-07-07                                   |26700     |
|588      |Huron, SD                    |South Dakota|2020-07-08                                   |26700     |
|590      |Huron, SD                    |South Dakota|2020-07-09                                   |26700     |


## Datasources
Source files that are used to make mapping files in `/src/optum_geo/data` are named by the <i>source_target resolution_year update</i> and are saved in their respective folders.

!!! danger ""
    **Warning:** As the census bureau updates these resolutions roughly once a year, ensure that you are using the most up to data mappings.


<!-- <div style="display: block; margin-right: auto; margin-left: auto;">  -->

|   Resolution     |   Source Link  |   Census Bureau  |
|---|----|----|
|   FIPS to CBSA   |   [source](https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html) |   Y    |
|   FIPS to CSA    |   [source](https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html ) |   Y   |
|   FIPS to State  |   [source](https://www.census.gov/geographies/reference-files/2017/demo/popest/2017-fips.html) (second file) |   Y   |
|   FIPS to MSA    |   [source](https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2020/delineation-files/list1_2020.xls ) |   Y  |
|   ZIP3 to FIPS   |   Optum Maps   |   N  |

<!-- </div> -->

---



## Installation

The optum-geo libary is very easy to get up and running. To do this you need to:

* Open your terminal 
* clone the [repo](https://github.com/uhg-internal/idp-geo-mapper).
* `cd ./optum-geo-standard`

Now you will need to install like any other pip installable package. (make sure you have activated your conda environment)

```bash
pip install .
```

Boom! now you have the **idp geo mapper** library installed 👍


## How to use

To use the package you import it like any other python package. 
```python
from optum_geo.aggregation import aggregator
```

Now you are ready to start converting your data. Here is an example below of how to convert a fips time series dataset into a CBSA dataset

```python
df, _ = aggregator.aggregate_data(data,
                                current_res = 'fips',
                                target_res = 'CBSA',
                                current_res_col = 'fips',
                                group_by = ['date'],
                                convert_columns = ['confirmed'], 
                                collect_unmappable = True,
                                mapping_file = None,
                                map_type = None,
                                mapping_src = 'census')
```


??? "aggregator.aggregate_data"
    Function takes in a data frame and converts the current geographical
    resolution to a higher one. Optum-geo uses data provided by the US census bureau for 
    it's mapping files, with the exception of mapping Zip3 to FIPS which is provided by the **Optum Maps** team. 

    **Args:**

    * `data (pd.DataFrame)`: the dataframe a user wishes to convert; be careful the given dataframe will be modified

    * `current_res (str)`: the current geographical resolution column the data is in
        
    * `target_res (str)`: the resolution the user wants to convert to [fips, zip3, CBSA, CSA]
        
    * `group_by (str)`: these columns will be used - in addition to `target_res` - to group 
        the dataframe rows and aggregate the `convert_columns`
    
    * `convert_columns (list)`: the columns they wish to aggregate [ex: `confirmed_cases`, `deaths` etc.]
        
    * `mapping_file`: if you have a file that is not included, it can be imported 
                    (target resolution must be lowercas and acronym only)
        
    * `map_type(str or None)`: only need to be specified if a custom `mapping_file` is used;
        must be `MAP_TYPE_FUNCTION` (a current res maps to at most one target) or
        `MAP_TYPE_ONE_TO_MANY` (a current res can map to multiple targets)
        
    * `collect_unmappable(bool)`: if set to `True`, maps everything that cannot be mapped to a `target_res` to `None` instead
        
    * `mapping_src (string)`: Choose the source from where the mapping files are generated [ex: census or optum_maps]
        
    **Returns:**
    
    * `pd.DataFrame`: result of aggregation in a dataframe and a dictionary containing meta information