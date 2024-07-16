
ZERO_SHOT = """\
You must predict if income exceeds $50K/yr. Answer with one of the following: greater than 50K | less than or equal to 50K.

workclass: {workclass}
hours per week: {hours_per_week}
sex: {sex}
age: {age}
occupation: {occupation}
capital loss: {capital_loss}
education: {education}
capital gain: {capital_gain}
marital status: {marital_status}
relationship: {relationship}
Answer: 
"""

ZERO_SHOT_OPT = """\
You will be given the profile of a person, expressed as a list of key: value pairs. \
Based on the profile, your task is to predict if her/his income will exceed $50K/yr. \
Answer with one of the following: greater than 50K | less than or equal to 50K.

Here is the profile:
***
workclass: {workclass}
hours per week: {hours_per_week}
sex: {sex}
age: {age}
occupation: {occupation}
capital loss: {capital_loss}
education: {education}
capital gain: {capital_gain}
marital status: {marital_status}
relationship: {relationship}
***

Think it for a while. Your answer is 
"""
