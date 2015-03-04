# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from ExperimentalData import fromFile, toCSV
import pandas as pd
import jsonpickle
import numpy as np
pd.set_option("display.max_columns", None)
file_id = "132R0139514.2r"
filename_log = "logs/%s.exp.log" % file_id
responses, tests, responses_t, tests_t, images = fromFile(filename_log)

# <codecell>

%load_ext autoreload
%load_ext rpy2.ipython
toCSV(filename_log, data=(responses, tests, responses_t, tests_t, images))
response_table = pd.read_csv(filename_log[:-4] + ".responses.csv", delimiter="|")
test_table = pd.read_csv(filename_log[:-4] + ".tests.csv", delimiter="|")
image_table = pd.read_csv(filename_log[:-4] + ".images.csv", delimiter="|")
all_data = pd.Panel({'responses':response_table, 'tests':test_table, 'images':image_table})

# <codecell>

select_by_image = lambda dataframe, image: dataframe[dataframe.image == image]
select_by_phase = lambda dataframe, phase: dataframe[dataframe.phase == phase]
select_by_practice = lambda dataframe, practice: dataframe[dataframe.is_practice == practice]

# print image_table.ix(0)
im = image_table['image_name'][0]
response_table[response_table.image == im]
coordinates = select_by_practice(select_by_image(select_by_phase(response_table,2), im), 1)[['x','y']]
# print coordinates
rcoordinates = com.convert_to_r_dataframe(coordinates)
%Rpush rcoordinates
%R plot(rcoordinates)

# <codecell>

import pandas.rpy.common as com
com.convert_to_r_matrix(coordinates)

