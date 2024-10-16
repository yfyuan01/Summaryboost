# KDD25_rebuttal
We sincerely thank all the reviewers for the valuable comments and suggestions they provided. We add our response as follows:

**Response to Reviewer1**

We sincerely thank the reviewer for the suggestions and below please find our response

$\underline{Response \space to \space the \space weaknesses}$

1. We thank the reviewer for their suggestion to change the naming of the task from "few-shot tabular classification" to "prompt-based tabular classification algorithms." We would like to clarify that our method emphasizes a training-free, LLM-based approach, particularly suited for low-resource, few-shot settings. We use the term "few-shot tabular classification" following terminology from related works [1, 2] to maintain consistency with existing literature. While we acknowledge the rich literature of research on self-supervised learning and Bayesian methods that achieve strong performance on benchmark datasets, our focus is on the advantages of LLMs in this context. Specifically, existing work [1] has already verified that LLMs excel in scenarios with **little** or **no** labeled training data, offering strong performance with minimal reliance on extensive training. The core benefit of LLM-based, prompt-driven approaches is their reduced training overhead, as performance can be optimized primarily through effective prompting strategies, saving substantial time in the classification process. Additionally, LLM-based methods demonstrate practical advantages in real-world applications, particularly by addressing bias and safety concerns without requiring full data visibility. We will revise the Introduction, Related Work, and Experimental Setup sections in our next version to clarify these points and further strengthen the paper.

> Thank you for your valuable suggestion regarding the positioning of our paper. We have endeavored to clarify our focus in the first sentence of the abstract, which states: "large language models (LLMs) for tabular classification in very few-shot scenarios". We term our task "few-shot tabular classification" to maintain consistency with existing literature [1, 2], and we apologize for any potential confusion this may have caused.  
Additionally, we would like to further elucidate that prompt-based tabular classification highlights LLM-based, training-free solutions, particularly suited for low-resource, few-shot settings. This type of solutions offers unique practical advantages, such as reduced development overhead and minimal data disclosure. Hence, they serve as a necessary complement to established self-supervised learning and Bayesian methods.  
In our next version, we will revise the Introduction, Related Work, and Experimental Setup sections to clarify these points and further strengthen the paper.

2. We thank the reviewer for the insightful feeback regarding the motivation of the prompting strategy. The motivation of such serialization consists several aspects and is supported by various previous works [1,2,3]: (1) while converting prompt-based inputs into structured formats for XGBoost is a valid approach, it introduces an additional step of transformation, potentially losing important contextual nuances that LLMs are designed to capture. (2) the design of combining XGBoost and LLM serialization involves factors such as flexibility and the ability of LLMs to generalize across tasks, as LLMs provide a unified framework that reduces the need for task-specific transformations. (3) This approach is more advantageous when few/no samples are provided, given LLMs' remarkable ability in terms of language understanding.

> We appreciate the reviewer's insightful inquiry regarding the trade-off between using LLMs for computation versus converting input for other models. Previous studies [1-2] have demonstrated that LLMs perform exceptionally well in **zero-** and **few-shot** tabular classification. Additionally, LLM-based approaches offer several practical benefits, as discussed earlier. These advantages are realized when structural input is serialized into natural language prompts.  
The serialization method, although appearing straightforward, is not trivial in two respects. First, the authors of TabLLM [1] extensively examined nine different serialization formats for tabular data and identified the "Text Template" format as performing well across various tasks (also mentioned in line 303 of our paper). Except for that, we have extended this approach into multifaceted serialization to incorporate least but necessary elements that help general-purpose LLMs perform specific tasks more effectively (line 313).  
Regarding XGBoost, its low dependence on feature pre-processing allows it to be robust across different domains and tasks, except for categorical features without a valid ordinal relationship. One-hot encoding is usually applied to these features; however, this method can be problematic for features with a long-tail value distribution. While we acknowledge the reviewer's suggestion that using LLMs to rewrite features could be a promising direction, this deserves independent and in-depth research and is beyond the scope of our current work.

   
3. Given the page and resource constraints of this paper, we respectfully argue that conducting experiments on the full dataset (30-40 instances) is challenging. Therefore, we follow previous works [1] and use the 9 most representative datasets with different scale to show the effectiveness of our method (the reason for selecting these datasets see our 3rd response in the next section).

> We understand the concern about potentially unreliable conclusions due to inadequate experimental validation. We would like to clarify that we are not asserting a general state-of-the-art performance over XGBoost, but rather demonstrating superior performance of our approach in a few-shot setting.  
There are two main considerations that guided our experimental design to support this claim. First, prompt-based tabular classification necessitates datasets with meaningful feature names and, ideally, descriptions. Second, the high computational and API costs associated with LLMs limit our ability to test on every possible dataset. Given these resource constraints, we followed previous work [1] and a recent survey [4], adopting the recommended nine datasets for our experiments.  
We have made every effort to ensure the reliability of our results. For instance, we employed five-fold cross-validation, tested on both open- and closed-source LLMs, and evaluated a comprehensive set of parameters. This represents one of the most extensive empirical studies to date in the field of prompt-based tabular classification. We respectfully hope you understand our situation and the choices we made. Thank you for your consideration.  
(Please also refer to our 3rd response in the next section for more details.)



$\underline{Response \space to \space the \space questions}$
1. While we agree that SSL-based methods can be applied to tabular classification, we emphasize that our focus is on the low-resource, few-shot setting where LLMs can be particularly effective. Bayesian algorithms, therefore, fall outside the scope of this work. However, for the reviewer's reference, we have included the results of TabPFN and XGBoost below (see **Table 1**). Our method demonstrates a clear advantage when applied to small training datasets, further validating its effectiveness in the targeted **few-shot** setting.

> We primarily claim SOTA performance in prompt-based tabular classification within low-resource, few-shot settings, which has been sufficiently verified through our existing tests and results. After considering the reviewer's comment, we agree that it is meaningful to also compare our approach with non-prompt methods. The results of TabPFN (a SSL-based approach) and XGBoost, as illustrated in **Table 1** of this rebuttal, show that our method demonstrates a clear advantage when applied to small training datasets. This further validates the effectiveness of our approach in the targeted **few-shot** setting.


2. We agree with the reviewer's concern about directly using XGBoost on each dataset. However, **since we are not the first to use LLMs for the tabular classificatio task, existing works have already made such comparison and verify the advantage of using LLM as a serialization tool against XGBoost (see Table 1 in [1])**. Therefore, we didn't report this result in the current version. However, we attach the result below. We will add the results in our next version.

2.5. the reasoing for using an LLM over XGBoost for our task has several key points: (1) LLMs are designed to understand complex relationships in data, in cases where the tabular data is represented in a natural language-like format (such as serialized descriptions or feature-rich text). As such, LLMs can exploit their pre-trained knowledge to extract deeper, contextual insights that a traditional algorithm like XGBoost might miss. (2) LLMs come with pre-trained knowledge that allows them to generalize across domains, this improves the model's adaptability and generatilization ability. (3) LLMs are especially good at learning in a low-resourse scenario, which allows for training-free few-shot classification.

> I do not understand P2: (2) LLMs provide a more unified approach and allows for end-to-end training. This reduces the complexity of the pipeline.

3. Due to budget constraints (particularly with closed-source LLMs like GPT) and time limitations, running the algorithm on the entire benchmark is challenging. Instead, we selected 9 datasets and evaluated 2 different LLMs to demonstrate the generalizability of our method. For dataset selection, we followed the approach outlined in [1, 4] to ensure a fair comparison. According to Section 4.1 in [1], the 9 datasets are selected for the following reasons: (1) datasets with at most 50,000 rows to keep the fine-tuning costs manageable and at most 30 columns to stay within T0’s token limit. (2) textual feature names are required to make the serializations more meaningful (3) datasets with derived feature values are excluded. 

4. The code is currently hosted in a private GitHub repository. While we are unable to release it at this time due to ~~safety concerns~~anonymity requirement, we have included the specific prompts used in our experiments in the paper to ensure reproducibility. We will release the code as soon as our paper gets accepted.

**We thank the reviewer again for the suggestions. We will add more elaborations and experimental results concerning the problems discussed. We sincerely hope the reviewer can kindly reassess the score if he/she finds our response helpful.**

*Table 1: F1 performance on the public tabular datasets with different number of training samples*
| num_examples | Method  | Bank | Blood | Calhous. | Car | Creditg | Diabetes | Heart | Income | Jungle | 
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|16     | XGBoost| 0.0| 0.273|0.664|0.332|0.801|0.312|0.693|0.105|0.528|
|16     | TabPFN| 0.013| 0.216|0.721|0.221|0.822|0.292|0.790|0.004|0.416|
|16     | **InsightTab**| 0.488|0.629|0.797|0.484|0.804|0.764|0.800|0.675|0.694|
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|32     | XGBoost|0.149|0.235|0.665|0.336|0.765|0.522|0.746|0.398|0.696|
|32     | TabPFN| 0.166| 0.095|0.760|0.238|0.818|0.372|0.798|0.086|0.701|
|32    | **InsightTab** |0.395|0.441|0.764|0.439|0.818|0.668|0.775|0.620|0.702|
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|64     | XGBoost|0.323|0.392|0.726|0.433|0.773|0.585|0.819|0.362|0.722|
|64     | TabPFN |0.214|0.116|0.800|0.285|0.804|0.577|0.837|0.350|0.723|
|64    | **InsightTab** |0.408|0.495|0.746|0.472|0.826|0.644|0.750|0.729|0.716|
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|128    | XGBoost|0.362|0.414|0.769|0.507|0.797|0.583|0.859|0.492|0.768|
|128    | TabPFN |0.180|0.278|0.832|0.477|0.806|0.625|0.848|0.368|0.772|
|128   | **InsightTab** |0.353|0.506|0.782|0.550|0.829|0.708|0.742|0.634|0.656|
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|256    | XGBoost|0.363|0.331|0.802|0.669|0.808|0.625|0.849|0.573|0.794|
|256    | TabPFN|0.272|0.280|0.842|0.674|0.812|0.642|0.866|0.486|0.787|
|256   | **InsightTab** |0.340|0.460|0.744|0.542|0.820|0.635|0.786|0.677|0.707|
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|All    | XGBoost|0.545|0.424|0.905| 0.977|0.828|0.609|0.876|0.724|0.891|
|All    | TabPFN|/|/|/|/|/|/|/|/|/|
|All   | **InsightTab** |0.368|0.476|0.716|0.476|0.837|0.655|0.756|0.642|0.708|


[1] TabLLM: Few-shot Classification of Tabular Data with Large Language Models

[2] TABLET: Learning From Instructions For Tabular Data

[3] Language models are weak learners. 

[4] Large Language Models (LLMs) on Tabular Data: Prediction, Generation, and Understanding - A Survey.


**Response to Reviewer2**

We sincerely thank the reviewer for the suggestions and below please find our response.

$\underline{Response \space to \space the \space weaknesses}$

1. We appreciate the reviewer's observation concerning the performance decreases with an increased number of few-shot samples. We agree that it is be a noteworthy aspect of our current framework. However, we believe this phenomenon does not necessarily indicate an inherent limitation of the framework in benefiting from larger datasets. Rather, it may be attributed to several factors: (1) This phenomenon can occur due to the inherent characteristics of LLM-based, prompt-driven methods, which may not benefit from larger data scales in the same way as traditional machine learning models. (2) as our setting is designed to simulate scenarios with very limited data, when more data is added incrementally, the framework might require adjustments (such as regularization or fine-tuning strategies) to fully leverage the larger sample size. We will add more clarification in our next version. 

> We appreciate the reviewer's insightful observation regarding the occasional decrease in performance with an increased number of samples. We agree that this is a noteworthy aspect of current LLM-based tabular classifiers that do not involve fine-tuning. We would like to further clarify that these approaches involve two types of samples: training samples and in-context samples (i.e., shots).  
**For training samples**: It appears that these approaches do not benefit from larger data scales in the same way as traditional machine learning models, as illustrated in Table 2 of our paper. Note that only a subset of training samples is selected as in-context samples for inference due to context window constraints. This could informally explain the unexpected trend where performance does not improve with more training samples.
**For in-context samples**: In-context samples have a direct impact on the LLM's predictions. We present the performance of tested approaches with different numbers of in-context samples in Figure 2 of our paper. The results indicate that, while other approaches still do not benefit from an increased number of shots, the performance of our InsightLLM continues to improve as the number of shots increases from 2 to 16. This is due to our strategic easy-first example selection and the distilled rules.  
In summary, this phenomenon arises from the inherent characteristics of LLM-based, prompt-driven methods. We have already optimized our method to leverage in-context samples effectively. Our current focus is on scenarios with very limited data. When more data becomes available incrementally, the framework may require adjustments, such as enabling fine-tuning, to fully utilize the larger sample size. We will add further clarification in our next version.  
Thank you for your valuable feedback.

2. We acknowledge that using a stronger LLM provides an advantage in rule summarization. To address the concern, we've added our experimental results with other rule LLMs below (see **Table 1**). We find our experiments with weaker LLMs like Mistral and gpt-4o-mini still demonstrate promising performance (also show SOTA performance on some datasets), as they retain the ability to generalize key patterns, even if their summarization capabilities are less robust. In addition, we agree that generalizing to more complex problems, where rules cannot be explicitly summarized, presents a challenge. Nevertheless, for harder tasks, we envision LLMs functioning in a more iterative mode, where they gather contextual information over multiple steps or turns from the selected examples, rather than relying solely on summarizing predefined rules. 

> Indeed, we deliberately distinguish between LLMs for rule summarization and prediction. Since rule summarization needs to be performed only once per dataset or task, this allows us to leverage the capabilities of more powerful LLMs in an economical manner, akin to a teacher-student collaboration.  
To address your concern, we have included further experimental results with other rule LLMs below (see **Table 1**). Our experiments with weaker LLMs such as Mistral and GPT-4o-mini still demonstrate promising performance, even achieving state-of-the-art results on some datasets. These models retain the ability to generalize key patterns, despite their less robust summarization capabilities.  
Additionally, we acknowledge that generalizing to more complex problems, where rules cannot be explicitly summarized, presents a challenge. For such tasks, we envision LLMs operating in a more iterative mode, gathering contextual information over multiple steps or turns from the selected examples, rather than relying solely on summarizing predefined rules.


$\underline{Response \space to \space the \space questions}$

Please refer to our responses to questions 1&2 by reviewer1 (vwUM).

**We thank the reviewer again for the suggestions. We will add more elaborations and experimental results concerning the problems discussed. We sincerely hope the reviewer can kindly reassess the score if he/she finds our response helpful.**

*Table 1: Results (gpt-3.5-turbo) of InsightTab based on gpt-4o-mini and Mistral generated rules.*
| num_examples | Rule Method  | Bank | Blood | Calhous. | Car | Creditg | Diabetes | Heart | Income | Jungle | 
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|16| gpt-4o-mini|0.367 | 0.395| 0.704 | 0.390 | 0.470 |0.656 | 0.703 |  0.599 | 0.617 |
|16| Mistral-7B| 0.372 | 0.429 | 0.702 | 0.477 | 0.786 | 0.643 | 0.722 | 0.584 | 0.626 |
|16|**gpt-4-turbo**| 0.417|  0.549 | 0.758|  0.515 | 0.809 | 0.682 | 0.767 | 0.610 | 0.649 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|32| gpt-4o-mini| 0.304 | 0.442| 0.709 |  0.429 |  0.429 | 0.635 | 0.750 | 0.620 | 0.603
|32| Mistral-7B| 0.374 | 0.456 | 0.715 | 0.480 | 0.793 | 0.645 | 0.741 | 0.601 | 0.630 |
|32 |**gpt-4-turbo**|0.358| 0.483 | 0.712| 0.491 | 0.814 | 0.646 | 0.788 | 0.647 | 0.636 | 
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|64| gpt-4o-mini|0.359 |0.458| 0.698| 0.467 | 0.502 | 0.638 | 0.774 | 0.617 | 0.631 |
|64| Mistral-7B|  0.368 | 0.477 | 0.701 | 0.492| 0.796 | 0.640  | 0.765 | 0.603 | 0.632 |
|64| **gpt-4-turbo**| 0.397|0.493| 0.697|  0.552| 0.835 | 0.633 | 0.812 | 0.620 | 0.636|
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|128| gpt-4o-mini| 0.366| 0.489 | 0.703 | 0.451 | 0.488 | 0.613 | 0.779 | 0.598 | 0.638 |
|128| Mistral-7B|  0.361 | 0.481 | 0.704 | 0.502 | 0.799 | 0.636 | 0.779 | 0.582 | 0.637 |
|128| **gpt-4-turbo**| 0.379| 0.502| 0.726 | 0.504 | 0.798 | 0.642 | 0.836 | 0.595 | 0.656 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|256| gpt-4o-mini| 0.357|   0.496| 0.708 | 0.509 | 0.515 | 0.618 | 0.787 | 0.593 | 0.619 |
|256| Mistral-7B|  0.355 | 0.484 | 0.698 | 0.503 | 0.788 | 0.610 | 0.792 | 0.590 | 0.627 |
|256| **gpt-4-turbo**| 0.390|0.514 | 0.706 | 0.522 | 0.808 | 0.616 | 0.818 | 0.685 | 0.644 |


**Response to Reviewer3**

We sincerely thank the reviewer for the suggestions and below please find our response

$\underline{Response \space to \space the \space weaknesses}$

1. We acknowledge that our framework does not explicitly incorporate the standard 'chain-of-thought' reasoning. However, we kindly argue that both the summarization and reflection mechanisms in our approach serve as forms of reasoning. Summarization allows the model to extract high-level, condensed rules from the dataset, which aids in better understanding the underlying patterns. Similarly, reflection functions as another type of reasoning, enabling the model to revisit and gain insights from samples that are prone to misclassification, thereby enhancing its ability to learn from mistakes. These mechanisms, while differing from traditional chain-of-thought reasoning, provide a powerful means for the model to process and analyze information in a structured way.

> 1. We adopt LLMs as the reasoning model and acknowledge that our framework does not explicitly incorporate the standard 'chain-of-thought' reasoning. However, we kindly argue that both the summarization and reflection mechanisms in our approach serve as forms of reasoning. Summarization allows the model to extract high-level, condensed rules from the dataset, which aids in better understanding the underlying patterns. Similarly, reflection functions as another type of reasoning, enabling the model to revisit and gain insights from samples that are prone to misclassification, thereby enhancing its ability to learn from mistakes. These mechanisms, while differing from traditional chain-of-thought reasoning, provide a powerful means for the model to process and analyze information in a structured way.

2. We quantify the "difficulty of prediction" by calculating the entropy generated by XGBoost for each data sample. Specifically, we rank the samples based on entropy scores, designating the $n_e$ samples with lowest entropy scores as easy and $n_h$ samples with highest entropy scores as difficult samples. For the reflection process, we test the prediction LLM by demonstrating easy samples and then applying it to hard samples. Misclassified hard samples are selected for further reflection, where the model learns a corrective rule to improve its performance.

> 2. We adopt a model-based grouping method rather than relying on predefined criteria. Specifically, we first fit an XGBoost model on the training set and group samples that belong to the same leaves in the first tree. Intuitively, samples within the same group should have similar key features (those on the path from root to leaves) and similar labels, thereby balancing the contributions of features and labels.  
We quantify the "difficulty of prediction" by calculating the entropy generated by XGBoost for each data sample. That is, an example is regarded as difficult if XGBoost fails to yield a confident prediction. Specifically, we rank the samples based on entropy scores, designating the $n_e$ samples with lowest entropy scores as easy and $n_h$ samples with highest entropy scores as difficult samples. For the reflection process, we test the prediction LLM by demonstrating easy samples and then applying it to hard samples. Misclassified hard samples are selected for further reflection, where the model learns a corrective rule to improve its performance.
   
3. We apologize for not reporting the computational cost for the proposed method due to page limit. Concerning the computation cost of the framework when the dataset has a large scale, while the three-step process of grouping, ranking, and summarizing might introduce some computational overhead, our design prioritizes efficiency and flexibility for the following reasons: (1) **Efficient Sampling**: instead of processing all data points, InsightTab can operate on representative samples, reducing the overall cost while still maintaining accuracy. (2) **Parallelization**: some steps of the framework can be parallized, which helps reduce the overall time consumption. For example, the grouping and ranking steps can be parallelized, making it feasible to scale InsightTab to larger datasets without significantly increasing computation time. (3) **Incremental Summarization**: for continuous or streaming data, InsightTab can summarize data incrementally rather than processing the entire dataset at once, reducing memory and computational burdens. (4) **Training-free solution**: although two LLMs are utilized in the framework, this does not imply that computational cost is a significant concern. Both LLMs are employed in a training-free manner, allowing for API-based methods that significantly reduce the overhead typically associated with model training. Therefore, we believe our method still shows advantage in terms of larger scale or more complicated scenarios. (We attach the overall cost below in **Table 1** to further address the reviewer's concern). We see our method shows advantage against other LLM prompt-based method (SummaryBoost as the most state-of-the-art) both in terms of time and budget cost. 

4. We acknowledge that the performance of our framework can be influenced by the limitations of the LLMs used. However, our goal is to reduce full dependency on LLMs by combining their strong language understanding with traditional machine learning techniques. Our approach leverages the strengths of LLMs in interpreting natural language descriptions of features, summarizing complex patterns, and serializing features into natural language sentences, without relying solely on them for computation. While LLMs may face challenges with intricate tabular data structures, our framework mitigates this by integrating the grouping, ranking, and summarization steps, which uncover relationships between features, even in complex tabular formats. To further assess InsightTab's versatility, we conducted experiments on 9 different tasks, demonstrating its ability to handle various data formats effectively.

5. Please kindly see our responses to reviewer1 (vwUM), specifically to Weakness 2 and Question 1 & 2.

*Table 1: Overall cost (USD) of InsightTab. Each number is reported using 5 fold validation on 128 training examples, 16 number of shots.*
|     | Bank     | Blood     | Calhous. | Car | Creditg | Diabetes | Heart | Income | Jungle | Time|
|---------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|Rule Generation (GPT-4o-mini)      | 0.02 |  0.03  | 0.01  | 0.02  | 0.01    | 0.01  |  0.02 |  0.02 |  0.01 | ~2min 20s|
|Rule Generation (GPT-4-turbo)      | 1.58 |  1.66 | 1.87 |  1.97 |  1.80   | 1.67  |  1.70 | 1.80  | 1.72  | ~2min 20s |
|Prediction  (GPT-3.5-turbo)   | 0.69 |  0.28  | 0.42 | 0.26  |   0.29  | 0.22  |  0.30 | 0.73  | 0.84  |~5min per 10k testing samples|
|SummaryBoost (GPT-4-turbo) | 12.3 | 14.5| 15.8| 14.1| 11.4| 10.4| 15.7|18.9 | 17.5 | ~20min per 10k testing samples|



**We thank the reviewer again for the suggestions. We will add more elaborations and experimental results concerning the problems discussed. We sincerely hope the reviewer can kindly reassess the score if he/she finds our response helpful.**

**Response to Reviewer4**

We sincerely thank the reviewer for the suggestions and below please find our response

$\underline{Response \space to \space the \space weaknesses}$

1. Please kindly see our response to reviewer1 (vwUM), specifically to Weakness 2 and Question 1 & 2.
   
2. We acknowledge the concern of rule example size, we will add a limitation section to discuss about it in detail. However, we propose two potential strategies to mitigate the issue: (1) Chunking and batching. Instead of inputting all examples at once, we can split the data into smaller chunks or batches that stay within the context length limit. The LLM can then summarize each chunk, and we can combine these partial summaries to generate a final rule set. (2) Sampling: For groups with a large number of examples, we can use representative sampling techniques to input a subset of the most informative examples, which reduces prompt size while still capturing the essential patterns needed for rule summarization. 

> 2. We acknowledge the concern of example explosion during rule summarization, and we will add a limitation section to discuss about it in detail. However, we propose two potential strategies to mitigate the issue: (1) Chunking and batching. Instead of inputting all examples at once, we can split the data into smaller chunks or batches that stay within the context length limit. The LLM can then summarize each chunk, and we can combine these partial summaries to generate a final rule set. (2) Sampling: For groups with a large number of examples, we can use representative sampling techniques to input a subset of the most informative examples, which reduces prompt size while still capturing the essential patterns needed for rule summarization. The sampling step will not incur too much information loss as data within groups are experted to be similar.

3. Concerning the cost of our model, please kindly see our response to weakness 3 by reviewer3 (5at8). We've also attached a table of the detailed cost. 

$\underline{Response \space to \space the \space questions}$

1. Please kindly see our response to reviewer1 (Response to Weakness 1 and Response to Question 2 & 2.5).

> 1. Please see our 1st response above.

2. Please refer to our overhead Table in response to reviewer3 (5at8).

**We thank the reviewer again for the suggestions. We will add more elaborations and experimental results concerning the problems discussed. We sincerely hope the reviewer can kindly reassess the score if he/she finds our response helpful.**

**Response to Reviewer5**

We sincerely thank the reviewer for the suggestions and below please find our response

$\underline{Response \space to \space the \space weaknesses}$

1. The advantage of our multi-step distillation process lies in enabling the model to gain a better understanding of the dataset through a few key examples and condensed rules. We respectfully argue that this approach minimizes the risk of error propagation. Unlike pipeline-based methods, where the generated rules may influence the selection of demonstrated samples (or vice versa), our approach separates the grouping and sampling processes with different methods. This independence ensures that errors in one step do not propagate to the other, allowing for a more robust and reliable distillation process. Additionally, the reflection mechanism in our approach enables the model to revisit and correct misclassified samples, providing an additional opportunity for error correction.
   
2. We acknowledge the concern of the influence of adopting different LLMs for rule summarization. Therefore, we present the experimental results with Mistral-generated and GPT-4o-mini generated rules (details and experimental results please see the response2 to reviewer2 UVNi).

$\underline{Response \space to \space the \space questions}$
1. Please see our 2nd response above.
   
2. Thank you for your observation. While the easy-first strategy is designed to select the most representative instances, its purpose is not to lead to convergence on a single rule, but rather to ensure that the model can generate rules that cover the most salient patterns in the data. In our approach, the easy-first strategy helps guide the model's understanding of the broader structure of the dataset, while subsequent instances can refine or introduce new patterns. Also the reflection on the hard samples enable the model to self-improve. The generation of a single rule is not very likely, as the framework is designed to handle multiple underlying patterns and adapt as more samples are introduced.

**We thank the reviewer again for the suggestions. We will add more elaborations and experimental results concerning the problems discussed. We sincerely hope the reviewer can kindly reassess the score if he/she finds our response helpful.**

