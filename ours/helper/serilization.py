# -*- coding: utf-8 -*-
# 
# Serilization methods for tabular records.
# 

# def serialize_by_tabllm(row, columns, question, answer_cond, usage):
#     # row = [f1, ..., fk, label]
#     note = " ".join([f"The {col} is {val}." for col, val in zip(columns[:-1], row[:-1]) if val is not None])
#     if usage == "train":
#         """ serilization for training examples
#         {description of the instance}

#         Does this client subscribe to a term deposit?
#         Answer: Yes
#         """
#         return (
#             f"{note}\n\n"
#             f"{question}\n"
#             f"Answer: {answer_cond[int(row[-1])]}"
#         )
#     elif usage == "test":
#         """ serilization for testing examples
#         {description of the instance}
        
#         Does this client subscribe to a term deposit? Answer the question with either Yes or No.
#         Answer: 
#         """
#         return (
#             f"{note}\n\n"
#             f"{question} {answer_cond}\n"
#             f"Answer: "
#         )
#     elif usage == "sv":
#         return (
#             f"[[label]]: {answer_cond[int(row[-1])]}\n"
#             f"[[feature]]: {note}"
#         ) 
#     else:
#         raise ValueError(f"encounter unknown usage={usage} in serialize_by_tabllm")

def serialize_by_tabllm(row:list, columns:list, question:str, answer_cond:list, is_train:bool):
    # row = [f1, ..., fk, label]
    note = " ".join([f"The {col} is {val}." for col, val in zip(columns[:-1], row[:-1]) if val is not None])
    if is_train is True:
        """ serilization for training examples
        {description of the instance}

        Does this client subscribe to a term deposit?
        Answer: Yes
        """
        return (
            f"{note}\n\n"
            f"{question}\n"
            f"Answer: {answer_cond[int(row[-1])]}"
        )
    else:
        """ serilization for testing examples
        {description of the instance}
        
        Does this client subscribe to a term deposit? Answer the question with either Yes or No.
        Answer:
        """
        return (
            f"{note}\n\n"
            f"{question} {answer_cond}\n"
            f"Answer: <xxx, Yes/No>"
        )