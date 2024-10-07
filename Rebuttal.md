# KDD25_rebuttal
We sincerely thank all the reviewers for the valuable comments and suggestions they provided. We add our response as follows:

**Response to Reviewer1**

We sincerely thank the reviewer for the suggestions and below please find our response

$\underline{Response \space to \space the \space weaknesses}$

1. We thank the reviewer for their suggestion to change the naming of the task from "few-shot tabular classification" to "prompt-based tabular classification algorithms." We would like to clarify that our method emphasizes a training-free, LLM-based approach, particularly suited for low-resource, few-shot settings. We use the term "few-shot tabular classification" following terminology from related works [1, 2] to maintain consistency with existing literature. While we acknowledge the rich literature of research on self-supervised learning and Bayesian methods that achieve strong performance on benchmark datasets, our focus is on the advantages of LLMs in this context. Specifically, existing work [1] has already verified that LLMs excel in scenarios with **little** or **no** labeled training data, offering strong performance with minimal reliance on extensive training. The core benefit of LLM-based, prompt-driven approaches is their reduced training overhead, as performance can be optimized primarily through effective prompting strategies, saving substantial time in the classification process. Additionally, LLM-based methods demonstrate practical advantages in real-world applications, particularly by addressing bias and safety concerns without requiring full data visibility. We will revise the Introduction, Related Work, and Experimental Setup sections in our next version to clarify these points and further strengthen the paper.

2. We thank the reviewer for the insightful feeback regarding the motivation of the prompting strategy. The motivation of such serialization consists several aspects and is supported by various previous works[1,2,3]: (1) while converting prompt-based inputs into structured formats for XGBoost is a valid approach, it introduces an additional step of transformation, potentially losing important contextual nuances that LLMs are designed to capture. (2) the design of combining XGBoost and LLM serialization involves factors such as flexibility and the ability of LLMs to generalize across tasks, as LLMs provide a unified framework that reduces the need for task-specific transformations. (3) This approach is more advantageous when few/no samples are provided, given LLMs' remarkable ability in terms of language understanding.
   
3.Given the page and resource constraints of this paper, we respectfully argue that conducting experiments on the full dataset (30-40 instances) is challenging. Therefore, we follow previous works [1] and use the 9 most representative datasets with different scale to show the effectiveness of our method (the reason for selecting these datasets see our 3rd response in the next section).

$\underline{Response \space to \space the \space questions}$
1. While we agree that SSL-based methods can be applied to tabular classification, we emphasize that our focus is on the low-resource, few-shot setting where LLMs can be particularly effective. Bayesian algorithms, therefore, fall outside the scope of this work. However, for the reviewer's reference, we have included the results of TabPFN below.

2. We agree with the reviewer's concern about directly using XGBoost on each dataset. However, **since we are not the first to use LLMs for the tabular classificatio task, existing works have already made such comparison and verify the advantage of using LLM as a serialization tool against XGBoost (see Table 1 in [1])**. Therefore, we didn't report this result for space constraints. However, we attach the result below. We will add the results in our next version.

2.5. the reasoing for using an LLM over XGBoost in our task has several key points: (1) LLMs are designed to understand complex relationships in data, in cases where the tabular data is represented in a natural language-like format (such as serialized descriptions or feature-rich text), LLMs can exploit their pre-trained knowledge to extract deeper, contextual insights that a traditional algorithm like XGBoost might miss. (2) LLMs provide a more unified approach and allows for end-to-end training. This reduces the complexity of the pipeline. (3) LLMs come with pre-trained knowledge that allows them to generalize across domains, this improves the model's adaptability and generatilization ability. (4) LLMs are especially good at learning in a low-resourse scenario, which allows for training-free few-shot classification.

3. Due to budget constraints (particularly with closed-source LLMs like GPT) and time limitations, running the algorithm on the entire benchmark is challenging. Instead, we selected 9 datasets and evaluated 2 different LLMs to demonstrate the generalizability of our method. For dataset selection, we followed the approach outlined in [1] to ensure a fair comparison. According to Section 4.1 in [1], the 9 datasets are selected for the following reasons: (1) datasets with at most 50,000 rows to keep the fine-tuning costs manageable and at most 30 columns to stay within T0’s token limit. (2) textual feature names are required to make the serializations more meaningful (3) datasets with derived feature values are excluded. 

4. The code is currently hosted in a private GitHub repository. While we are unable to release it at this time due to safety concerns, we have included the specific prompts used in our experiments in the paper to ensure reproducibility. We will release the code as soon as our paper gets accepted.

**We thank the reviewer again for the suggestions. We will add more elaborations and experimental results concerning the problems discussed. We sincerely hope the reviewer can consider these revisions during the rebuttal phase and kindly reassess the overall scoren.**

[1] TabLLM: Few-shot Classification of Tabular Data with Large Language Models
[2] TABLET: Learning From Instructions For Tabular Data
[3] Language models are weak learners. 


**Response to Reviewer2**

We sincerely thank the reviewer for the suggestions and below please find our response

$\underline{Response \space to \space the \space weaknesses}$

1. We appreciate the reviewer's observation concerning the performance decrease with an increased number of few-shot samples. We agree that it is be a noteworthy aspect of our current framework. However, we believe this phenomenon does not necessarily indicate an inherent limitation of the framework in benefiting from larger datasets. Rather, it may be attributed to several factors: (1) This phenomenon can occur due to the inherent characteristics of LLM-based, prompt-driven methods, which may not benefit from larger data scales in the same way as traditional machine learning models. (2) as our setting is designed to simulate scenarios with very limited data, when more data is added incrementally, the framework might require adjustments (such as regularization or fine-tuning strategies) to fully leverage the larger sample size. We will add more clarification in our next version. 

2. We acknowledge that using a stronger LLM provides an advantage in rule summarization. However, extracting rules from a strong LLM is often straightforward and does not require extra training or much prompting. Also, our experiments with weaker open-sourced LLMs like Mistral still demonstrate promising performance (see the result below), as they retain the ability to generalize key patterns, even if their summarization capabilities are less robust. In addition, we agree that generalizing to more complex problems, where rules cannot be explicitly summarized, presents a challenge. Nevertheless, for harder tasks, we envision LLMs functioning in a more iterative mode, where they gather contextual information over multiple steps or turns from the selected examples, rather than relying solely on summarizing predefined rules. 

$\underline{Response \space to \space the \space questions}$

Please see response 2 and 2.5 to the questions from reviewer1 for more details. 

**We thank the reviewer again for the suggestions. We will add more elaborations and experimental results concerning the problems discussed. We sincerely hope the reviewer can consider these revisions during the rebuttal phase and kindly reassess the overall scoren.**

**Response to Reviewer4**
