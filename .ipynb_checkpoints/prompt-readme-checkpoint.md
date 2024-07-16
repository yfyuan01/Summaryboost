# task-related info has been used somewhere currently
```json
{
    "bank": {
        "description": "You are dealing with a problem of predicting whether a customer subscribes to a term deposit or not based on the following features.\n# bank client features:\n1 - age (numeric)\n2 - type of job (categorical: \"admin.\", \"unknown\", \"unemployed\", \"management\", \"housemaid\", \"entrepreneur\", \"student\", \"blue-collar\", \"self-employed\", \"retired\", \"technician\", \"services\") \n3 - marital status (categorical: \"married\",\"divorced\",\"single\"; note: \"divorced\" means divorced or widowed)\n4 - education (categorical: \"unknown\", \"secondary\", \"primary\", \"tertiary\")\n5 - has credit in default? (binary: \"yes\", \"no\")\n6 - average yearly balance, in euros (numeric) \n7 - has housing loan? (binary: \"yes\", \"no\")\n8 - has personal loan? (binary: \"yes\", \"no\")\n# related with the last contact of the current campaign:\n9 - contact communication type (categorical: \"unknown\", \"telephone\", \"cellular\") \n10 - last contact day of the month (numeric)\n11 - last contact month of year (categorical: \"jan\", \"feb\", \"mar\", ..., \"nov\", \"dec\")\n12 - last contact duration, in seconds (numeric)\n# other attributes:\n13 - number of contacts performed during this campaign and for this client (numeric, includes last contact)\n14 - number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)\n15 - number of contacts performed before this campaign and for this client (numeric)\n16 - outcome of the previous marketing campaign (categorical: \"unknown\", \"other\", \"failure\", \"success\")",
        "title": "Title: Deposit subscription prediction",
        "description_brief": "This data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. The classification goal is to predict if the client will subscribe a term deposit.",
        "question": "Does this client subscribe to a term deposit?",
        "answer_requirement_brief": "Answer the question with either Yes or No.",
        "answer_requirement": "Answer the question with either 'Yes' or 'No' (without quotes).",
        "answer_choices": [
            "No",
            "Yes"
        ],
        "trend": "Please distill the key trends that may assist an AI model in making future predictions. Output trends only without any further explanations.",
        "summarization": "Tl;dr / Summarize the rules into a small set of non-conflicting and complementary patterns for predicting whether a client would subscribe to a term deposit. Output patterns only without any further explanations."
    }, 
    # omitting other datasets
}
```


# Base prompt by Yifei 

Base prompt means only leveraging the task-related info at hand, and advanced prompts such as by summaryboost and cocktail could further incorporate other information to base prompt.

## Meta-template

```
{dataset.title}
{dataset.description_brief}

###

Examples:

{dataset.example_1.serialization}
{dataset.question}
Answer: {dataset.answer_choices[dataset.example_1.label]}

---

OTHER_EXAMPLES...

###

Now here's the question: {dataset.example_1.serialization}
{dataset.question} {dataset.answer_requirement_brief}
Answer: 
```

## An example on bank

Please refer to the task information defined in the beginning json. 

```
Title: Deposit subscription prediction
This data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. The classification goal is to predict if the client will subscribe a term deposit.

###

Examples:

The age is 42. The type of job is admin.. The marital status is single. The education is secondary. The has credit in default? is no. The average yearly balance, in euros is 4343. The has housing loan? is no. The has personal loan? is no. The contact communication type is cellular. The last contact day of the month is 31. The last contact month of year is jul. The last contact duration, in seconds is 245. The number of contacts performed during this campaign and for this client is 4. The number of days that passed by after the client was last contacted from a previous campaign is client was not previously contacted. The number of contacts performed before this campaign and for this client is 0. The outcome of the previous marketing campaign is unknown.
Does this client subscribe to a term deposit?
Answer: No

---

omitting other examples...

###

Now here's the question: The age is 31. The type of job is admin.. The marital status is single. The education is tertiary. The has credit in default? is no. The average yearly balance, in euros is 2004. The has housing loan? is no. The has personal loan? is no. The contact communication type is cellular. The last contact day of the month is 19. The last contact month of year is nov. The last contact duration, in seconds is 117. The number of contacts performed during this campaign and for this client is 1. The number of days that passed by after the client was last contacted from a previous campaign is client was not previously contacted. The number of contacts performed before this campaign and for this client is 0. The outcome of the previous marketing campaign is unknown.
Does this client subscribe to a term deposit? Answer the question with either 'Yes' or 'No' (without quotes).
Answer: 
``` 

# Base prompt by Jiatong

## Meta-prompt 

```
PATTERNS MINED (a unique part that will not be used in base prompt) 

----------------

{dataset.description}

----------------

[FEW-SHOT EXAMPLES START]

----------------

{dataset.example_1.serialization}

{dataset.question}
Answer: {dataset.answer_choices[dataset.example_1.label]}

----------------

OTHER_EXAMPLES...

----------------

[FEW-SHOT EXAMPLES END]

----------------

[CURRENT QUESTION START]

----------------

{dataset.example_1.serialization}

{dataset.question} {dataset.answer_requirement}
Answer: <xxx, {dataset.answer_choices}>

```

## An example on bank 

```
Useful patterns for the task at hand:
1. Clients with no housing loan are more likely to subscribe to a term deposit.
2. Clients with no personal loan are more likely to subscribe to a term deposit.
3. A higher average yearly balance correlates with a higher likelihood of subscribing to a term deposit.
4. Longer last contact durations generally increase the likelihood of a client subscribing to a term deposit.
5. Clients who were successfully contacted or had a positive outcome in previous marketing campaigns are more likely to subscribe to a term deposit.
6. Clients who have not been previously contacted or for whom the number of days since last contact from a previous campaign is high tend to have a higher likelihood of subscribing.
7. Clients contacted via cellular phone show a higher subscription rate to term deposits.
8. The presence of a personal loan does not show a consistent trend in affecting term deposit subscriptions.
9. Marital status, education level, and job type do not show a clear influence on the likelihood of subscribing to a term deposit.

----------------

You are dealing with a problem of predicting whether a customer subscribes to a term deposit or not based on the following features.
# bank client features:
1 - age (numeric)
2 - type of job (categorical: "admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services") 
3 - marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
4 - education (categorical: "unknown", "secondary", "primary", "tertiary")
5 - has credit in default? (binary: "yes", "no")
6 - average yearly balance, in euros (numeric) 
7 - has housing loan? (binary: "yes", "no")
8 - has personal loan? (binary: "yes", "no")
# related with the last contact of the current campaign:
9 - contact communication type (categorical: "unknown", "telephone", "cellular") 
10 - last contact day of the month (numeric)
11 - last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
12 - last contact duration, in seconds (numeric)
# other attributes:
13 - number of contacts performed during this campaign and for this client (numeric, includes last contact)
14 - number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
15 - number of contacts performed before this campaign and for this client (numeric)
16 - outcome of the previous marketing campaign (categorical: "unknown", "other", "failure", "success")

----------------

[FEW-SHOT EXAMPLES START]

----------------

The age is 28. The type of job is services. The marital status is single. The education is primary. The has credit in default? is no. The average yearly balance, in euros is 307. The has housing loan? is no. The has personal loan? is no. The contact communication type is cellular. The last contact day of the month is 30. The last contact month of year is apr. The last contact duration, in seconds is 132. The number of contacts performed during this campaign and for this client is 1. The number of days that passed by after the client was last contacted from a previous campaign is client was not previously contacted. The number of contacts performed before this campaign and for this client is 0. The outcome of the previous marketing campaign is unknown.

Does this client subscribe to a term deposit?
Answer: No

----------------

omittint other examples...

----------------

[FEW-SHOT EXAMPLES END]

----------------

[CURRENT QUESTION START]

----------------

The age is 28. The type of job is blue-collar. The marital status is single. The education is secondary. The has credit in default? is no. The average yearly balance, in euros is 623. The has housing loan? is no. The has personal loan? is no. The contact communication type is unknown. The last contact day of the month is 18. The last contact month of year is jun. The last contact duration, in seconds is 25. The number of contacts performed during this campaign and for this client is 41. The number of days that passed by after the client was last contacted from a previous campaign is client was not previously contacted. The number of contacts performed before this campaign and for this client is 0. The outcome of the previous marketing campaign is unknown.

Does this client subscribe to a term deposit? Answer the question with either 'Yes' or 'No' (without quotes).
Answer: <xxx, Yes/No>"
```


# Final base prompt

- Use yifei's title and brief description as background

- Follow Jiatong's more explicit splitter, while removing some -------- lines

- Use Jiatong's answer requirement and format

## Meta prompt
```
{dataset.title}
{dataset.description_brief}

###

[FEW-SHOT EXAMPLES START]

{dataset.example_1.serialization}

{dataset.question}
Answer: {dataset.answer_choices[dataset.example_1.label]}

---

OTHER_EXAMPLES...

[FEW-SHOT EXAMPLES END]

###

[CURRENT QUESTION START]

Now here's the question: {dataset.example_1.serialization}

{dataset.question} {dataset.answer_requirement}
Answer: <xxx, {dataset.answer_choices}>

```

## A complete example on bank

```
Title: Deposit subscription prediction
This data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. The classification goal is to predict if the client will subscribe a term deposit.

###

[FEW-SHOT EXAMPLES START]

The age is 42. The type of job is management. The marital status is divorced. The education is secondary. The has credit in default? is no. The average yearly balance, in euros is 830. The has housing loan? is yes. The has personal loan? is no. The contact communication type is unknown. The last contact day of the month is 19. The last contact month of year is may. The last contact duration, in seconds is 112. The number of contacts performed during this campaign and for this client is 1. The number of days that passed by after the client was last contacted from a previous campaign is client was not previously contacted. The number of contacts performed before this campaign and for this client is 0. The outcome of the previous marketing campaign is unknown.

Does this client subscribe to a term deposit?
Answer: No

---

The age is 47. The type of job is blue-collar. The marital status is married. The education is secondary. The has credit in default? is no. The average yearly balance, in euros is -98. The has housing loan? is yes. The has personal loan? is no. The contact communication type is unknown. The last contact day of the month is 9. The last contact month of year is may. The last contact duration, in seconds is 112. The number of contacts performed during this campaign and for this client is 1. The number of days that passed by after the client was last contacted from a previous campaign is client was not previously contacted. The number of contacts performed before this campaign and for this client is 0. The outcome of the previous marketing campaign is unknown.

Does this client subscribe to a term deposit?
Answer: No

---

The age is 41. The type of job is blue-collar. The marital status is divorced. The education is secondary. The has credit in default? is no. The average yearly balance, in euros is 14. The has housing loan? is yes. The has personal loan? is yes. The contact communication type is unknown. The last contact day of the month is 13. The last contact month of year is may. The last contact duration, in seconds is 65. The number of contacts performed during this campaign and for this client is 6. The number of days that passed by after the client was last contacted from a previous campaign is client was not previously contacted. The number of contacts performed before this campaign and for this client is 0. The outcome of the previous marketing campaign is unknown.

Does this client subscribe to a term deposit?
Answer: No

---

The age is 38. The type of job is technician. The marital status is single. The education is secondary. The has credit in default? is no. The average yearly balance, in euros is 1270. The has housing loan? is yes. The has personal loan? is yes. The contact communication type is cellular. The last contact day of the month is 15. The last contact month of year is may. The last contact duration, in seconds is 164. The number of contacts performed during this campaign and for this client is 1. The number of days that passed by after the client was last contacted from a previous campaign is 337. The number of contacts performed before this campaign and for this client is 6. The outcome of the previous marketing campaign is other.

Does this client subscribe to a term deposit?
Answer: No

---

The age is 69. The type of job is retired. The marital status is married. The education is unknown. The has credit in default? is no. The average yearly balance, in euros is 426. The has housing loan? is no. The has personal loan? is no. The contact communication type is telephone. The last contact day of the month is 9. The last contact month of year is dec. The last contact duration, in seconds is 395. The number of contacts performed during this campaign and for this client is 1. The number of days that passed by after the client was last contacted from a previous campaign is client was not previously contacted. The number of contacts performed before this campaign and for this client is 0. The outcome of the previous marketing campaign is unknown.

Does this client subscribe to a term deposit?
Answer: Yes

---

The age is 33. The type of job is management. The marital status is married. The education is tertiary. The has credit in default? is no. The average yearly balance, in euros is 2441. The has housing loan? is yes. The has personal loan? is yes. The contact communication type is cellular. The last contact day of the month is 19. The last contact month of year is nov. The last contact duration, in seconds is 154. The number of contacts performed during this campaign and for this client is 1. The number of days that passed by after the client was last contacted from a previous campaign is client was not previously contacted. The number of contacts performed before this campaign and for this client is 0. The outcome of the previous marketing campaign is unknown.

Does this client subscribe to a term deposit?
Answer: No

---

The age is 42. The type of job is blue-collar. The marital status is married. The education is primary. The has credit in default? is no. The average yearly balance, in euros is 2519. The has housing loan? is yes. The has personal loan? is no. The contact communication type is unknown. The last contact day of the month is 15. The last contact month of year is may. The last contact duration, in seconds is 262. The number of contacts performed during this campaign and for this client is 4. The number of days that passed by after the client was last contacted from a previous campaign is client was not previously contacted. The number of contacts performed before this campaign and for this client is 0. The outcome of the previous marketing campaign is unknown.

Does this client subscribe to a term deposit?
Answer: No

---

The age is 40. The type of job is blue-collar. The marital status is married. The education is secondary. The has credit in default? is no. The average yearly balance, in euros is 1844. The has housing loan? is yes. The has personal loan? is no. The contact communication type is cellular. The last contact day of the month is 11. The last contact month of year is may. The last contact duration, in seconds is 231. The number of contacts performed during this campaign and for this client is 1. The number of days that passed by after the client was last contacted from a previous campaign is client was not previously contacted. The number of contacts performed before this campaign and for this client is 0. The outcome of the previous marketing campaign is unknown.

Does this client subscribe to a term deposit?
Answer: No

---

The age is 54. The type of job is blue-collar. The marital status is married. The education is secondary. The has credit in default? is no. The average yearly balance, in euros is 1765. The has housing loan? is yes. The has personal loan? is no. The contact communication type is unknown. The last contact day of the month is 14. The last contact month of year is may. The last contact duration, in seconds is 188. The number of contacts performed during this campaign and for this client is 1. The number of days that passed by after the client was last contacted from a previous campaign is client was not previously contacted. The number of contacts performed before this campaign and for this client is 0. The outcome of the previous marketing campaign is unknown.

Does this client subscribe to a term deposit?
Answer: No

---

The age is 51. The type of job is entrepreneur. The marital status is married. The education is tertiary. The has credit in default? is no. The average yearly balance, in euros is 83. The has housing loan? is no. The has personal loan? is no. The contact communication type is cellular. The last contact day of the month is 20. The last contact month of year is apr. The last contact duration, in seconds is 267. The number of contacts performed during this campaign and for this client is 2. The number of days that passed by after the client was last contacted from a previous campaign is client was not previously contacted. The number of contacts performed before this campaign and for this client is 0. The outcome of the previous marketing campaign is unknown.

Does this client subscribe to a term deposit?
Answer: Yes

---

The age is 29. The type of job is blue-collar. The marital status is married. The education is secondary. The has credit in default? is no. The average yearly balance, in euros is 912. The has housing loan? is yes. The has personal loan? is no. The contact communication type is cellular. The last contact day of the month is 13. The last contact month of year is may. The last contact duration, in seconds is 785. The number of contacts performed during this campaign and for this client is 1. The number of days that passed by after the client was last contacted from a previous campaign is client was not previously contacted. The number of contacts performed before this campaign and for this client is 0. The outcome of the previous marketing campaign is unknown.

Does this client subscribe to a term deposit?
Answer: No

---

The age is 31. The type of job is technician. The marital status is single. The education is tertiary. The has credit in default? is no. The average yearly balance, in euros is 36. The has housing loan? is no. The has personal loan? is no. The contact communication type is cellular. The last contact day of the month is 12. The last contact month of year is aug. The last contact duration, in seconds is 111. The number of contacts performed during this campaign and for this client is 3. The number of days that passed by after the client was last contacted from a previous campaign is client was not previously contacted. The number of contacts performed before this campaign and for this client is 0. The outcome of the previous marketing campaign is unknown.

Does this client subscribe to a term deposit?
Answer: No

---

The age is 59. The type of job is retired. The marital status is married. The education is primary. The has credit in default? is no. The average yearly balance, in euros is 4450. The has housing loan? is no. The has personal loan? is no. The contact communication type is cellular. The last contact day of the month is 28. The last contact month of year is aug. The last contact duration, in seconds is 106. The number of contacts performed during this campaign and for this client is 3. The number of days that passed by after the client was last contacted from a previous campaign is client was not previously contacted. The number of contacts performed before this campaign and for this client is 0. The outcome of the previous marketing campaign is unknown.

Does this client subscribe to a term deposit?
Answer: No

---

The age is 34. The type of job is management. The marital status is single. The education is tertiary. The has credit in default? is no. The average yearly balance, in euros is 454. The has housing loan? is no. The has personal loan? is no. The contact communication type is cellular. The last contact day of the month is 14. The last contact month of year is aug. The last contact duration, in seconds is 76. The number of contacts performed during this campaign and for this client is 3. The number of days that passed by after the client was last contacted from a previous campaign is client was not previously contacted. The number of contacts performed before this campaign and for this client is 0. The outcome of the previous marketing campaign is unknown.

Does this client subscribe to a term deposit?
Answer: No

---

The age is 86. The type of job is retired. The marital status is married. The education is secondary. The has credit in default? is no. The average yearly balance, in euros is 1503. The has housing loan? is no. The has personal loan? is no. The contact communication type is telephone. The last contact day of the month is 18. The last contact month of year is mar. The last contact duration, in seconds is 165. The number of contacts performed during this campaign and for this client is 3. The number of days that passed by after the client was last contacted from a previous campaign is 101. The number of contacts performed before this campaign and for this client is 1. The outcome of the previous marketing campaign is other.

Does this client subscribe to a term deposit?
Answer: No

---

The age is 32. The type of job is blue-collar. The marital status is single. The education is primary. The has credit in default? is no. The average yearly balance, in euros is 181. The has housing loan? is yes. The has personal loan? is no. The contact communication type is cellular. The last contact day of the month is 6. The last contact month of year is may. The last contact duration, in seconds is 90. The number of contacts performed during this campaign and for this client is 1. The number of days that passed by after the client was last contacted from a previous campaign is client was not previously contacted. The number of contacts performed before this campaign and for this client is 0. The outcome of the previous marketing campaign is unknown.

Does this client subscribe to a term deposit?
Answer: No

[FEW-SHOT EXAMPLES END]

###

[CURRENT QUESTION START]

Now here's the question: The age is 28. The type of job is blue-collar. The marital status is single. The education is secondary. The has credit in default? is no. The average yearly balance, in euros is 623. The has housing loan? is no. The has personal loan? is no. The contact communication type is unknown. The last contact day of the month is 18. The last contact month of year is jun. The last contact duration, in seconds is 25. The number of contacts performed during this campaign and for this client is 41. The number of days that passed by after the client was last contacted from a previous campaign is client was not previously contacted. The number of contacts performed before this campaign and for this client is 0. The outcome of the previous marketing campaign is unknown.

Does this client subscribe to a term deposit? Answer the question with either 'Yes' or 'No' (without quotes).
Answer: <xxx, Yes/No>
```

## A complete example on car

For car dataset, the last line is as follows: 

Answer: <xxx, Unacceptable/Acceptable/Good/Very Good>

```
Title: Car safety prediction
This dataset was derived from a simple hierarchical decision model originally developed for the demonstration of DEX. The goal is to evaluate the safety of cars.

###

[FEW-SHOT EXAMPLES START]

The Buying price is medium. The Doors is four. The Maintenance costs is low. The Persons is more than four. The Safety score is high. The Trunk size is big.

How would you rate the decision to buy this car?
Answer: Very Good

---

The Buying price is low. The Doors is four. The Maintenance costs is high. The Persons is two. The Safety score is medium. The Trunk size is medium.

How would you rate the decision to buy this car?
Answer: Unacceptable

---

The Buying price is medium. The Doors is three. The Maintenance costs is medium. The Persons is four. The Safety score is low. The Trunk size is medium.

How would you rate the decision to buy this car?
Answer: Unacceptable

---

The Buying price is very high. The Doors is three. The Maintenance costs is low. The Persons is two. The Safety score is high. The Trunk size is big.

How would you rate the decision to buy this car?
Answer: Unacceptable

---

The Buying price is very high. The Doors is two. The Maintenance costs is low. The Persons is two. The Safety score is high. The Trunk size is medium.

How would you rate the decision to buy this car?
Answer: Unacceptable

---

The Buying price is low. The Doors is three. The Maintenance costs is low. The Persons is more than four. The Safety score is high. The Trunk size is medium.

How would you rate the decision to buy this car?
Answer: Very Good

---

The Buying price is low. The Doors is four. The Maintenance costs is medium. The Persons is more than four. The Safety score is medium. The Trunk size is medium.

How would you rate the decision to buy this car?
Answer: Good

---

The Buying price is medium. The Doors is three. The Maintenance costs is very high. The Persons is two. The Safety score is medium. The Trunk size is small.

How would you rate the decision to buy this car?
Answer: Unacceptable

---

The Buying price is low. The Doors is two. The Maintenance costs is low. The Persons is more than four. The Safety score is medium. The Trunk size is small.

How would you rate the decision to buy this car?
Answer: Unacceptable

---

The Buying price is medium. The Doors is two. The Maintenance costs is high. The Persons is more than four. The Safety score is medium. The Trunk size is big.

How would you rate the decision to buy this car?
Answer: Acceptable

---

The Buying price is high. The Doors is three. The Maintenance costs is medium. The Persons is two. The Safety score is low. The Trunk size is medium.

How would you rate the decision to buy this car?
Answer: Unacceptable

---

The Buying price is low. The Doors is two. The Maintenance costs is medium. The Persons is more than four. The Safety score is high. The Trunk size is small.

How would you rate the decision to buy this car?
Answer: Unacceptable

---

The Buying price is medium. The Doors is four. The Maintenance costs is medium. The Persons is two. The Safety score is low. The Trunk size is big.

How would you rate the decision to buy this car?
Answer: Unacceptable

---

The Buying price is medium. The Doors is three. The Maintenance costs is very high. The Persons is two. The Safety score is high. The Trunk size is big.

How would you rate the decision to buy this car?
Answer: Unacceptable

---

The Buying price is very high. The Doors is three. The Maintenance costs is low. The Persons is two. The Safety score is high. The Trunk size is medium.

How would you rate the decision to buy this car?
Answer: Unacceptable

---

The Buying price is low. The Doors is four. The Maintenance costs is low. The Persons is four. The Safety score is low. The Trunk size is medium.

How would you rate the decision to buy this car?
Answer: Unacceptable

[FEW-SHOT EXAMPLES END]

###

[CURRENT QUESTION START]

Now here's the question: The Buying price is high. The Doors is four. The Maintenance costs is very high. The Persons is more than four. The Safety score is high. The Trunk size is small.

How would you rate the decision to buy this car? Answer the question with either 'Unacceptable', 'Acceptable', 'Good', or 'Very Good' (without quotes).
Answer: <xxx, Unacceptable/Acceptable/Good/Very Good>
```
