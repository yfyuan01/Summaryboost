# KDD25_rebuttal
We sincerely thank all the reviewers for the valuable comments and suggestions they provided. We add our response as follows:

**Response to Reviewer1**

$\underline{Response \space to \space the \space weaknesses}$

We sincerely thank the reviewer for the suggestions and below please find our response

1. We thank the reviewer for their suggestion to change the naming of the task from "few-shot tabular classification" to "prompt-based tabular classification algorithms." We would like to clarify that our method emphasizes a training-free, LLM-based approach, particularly suited for low-resource, few-shot settings. We use the term "few-shot tabular classification" following terminology from related works [1, 2] to maintain consistency with existing literature. While we acknowledge the rich literature of research on self-supervised learning and Bayesian methods that achieve strong performance on benchmark datasets, our focus is on the advantages of LLMs in this context. Specifically, existing work [1] has already verified that LLMs excel in scenarios with **little** or **no** labeled training data, offering strong performance with minimal reliance on extensive training. The core benefit of LLM-based, prompt-driven approaches is their reduced training overhead, as performance can be optimized primarily through effective prompting strategies, saving substantial time in the classification process. Additionally, LLM-based methods demonstrate practical advantages in real-world applications, particularly by addressing bias and safety concerns without requiring full data visibility. We will revise the Introduction, Related Work, and Experimental Setup sections in our next version to clarify these points and further strengthen the paper.

2. We
3.

$\underline{Response \space to \space the \space questions}$
1. In this case, Bayesian algorithms are beyond our scope. However, we attach the result of TabPFN below for the reviewer to have a reference. 

2. We agree with your concern abobut directly using XGBoost on each dataset. Actually, existing works have already made such comparison and verify the advantage of using LLM as a serialization tool against XGBoost (see Table 1 in [1]). Therefore, we didn't report this result for space limitation. However, we attach the result below. We will add the results in our next version.

2.5

3. Concerning the budget (on closed-source LLMs e.g. GPT) and time limit, it's a tricky to perform the algorithm on the full benchamark. We pick the 9 datasets and investigate 2 different LLMs to demonstrate the generalizibity of our method. For the dataset selection, we follow existing work [1], to make a fair comparison against it. According to Section 4.1 in [1], the 9 datasets are selected for the following reasons: (1) datasets with at most 50,000 rows to keep the fine-tuning costs manageable and at most 30 columns to stay within T0’s token limit. (2) textual feature names are required to make the serializations more meaningful (3) datasets with derived feature values are excluded. 

4. The code is currently hosted in a private GitHub repository. While we are unable to release it at this time due to safety concerns, we have included the specific prompts used in our experiments in the paper to ensure reproducibility. We will release the code as soon as our paper gets accepted.

We thank again for the time the reviewer spent on our paper. We will improve the paper according to all the issues the reviewer mentioned in our next version.

[1] TabLLM: Few-shot Classification of Tabular Data with Large Language Models
[2] TABLET: Learning From Instructions For Tabular Data
