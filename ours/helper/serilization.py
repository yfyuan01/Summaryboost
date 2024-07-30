# -*- coding: utf-8 -*-
# 
# Serilization methods for tabular records.
# 


def serialize_by_tabllm(row:list, columns:list, question:str, answer_cond:list, is_train:bool, answer_requirement:str = None):
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
        answer_options = '/'.join(answer_cond)
        return (
            f"{note}\n\n"
            f"{question} {answer_requirement}\n"
            f"Answer: <xxx, {answer_options}>"
        )