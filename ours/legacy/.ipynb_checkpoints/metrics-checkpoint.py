# -*- coding: utf-8 -*-
# Author: renjun.hrj
# 2023-12-25
#
# Tabular prediction metrics:
#   Effectiveness: accuracy, F1
#   Fairness: SPD (statistical parity difference), EOD (equality of opportunity)
#


def income_pred_label(output):
    labels = []
    if "greater than 50K" in output:
        labels.append(1)
    if "the predicted income exceeds $50K/yr" in output:
        labels.append(1)
    if "less than or equal to 50K" in output:
        labels.append(0)
    if "less than or equal to $50K" in output:
        labels.append(0)
    if "Less than or equal to 50K" in output:
        labels.append(0)
    if not labels or 1 in labels and 0 in labels:
        return None 
    return labels[0]


def calc_metrics(dataset, examples, gender=None):
    if gender is not None:
        examples = [example for example in examples if example["sex"] == gender]
    total = len(examples)

    if dataset == "income":    
        target = set(example["target"] for example in examples)
        print("Targets:", target)
        gender = set(example["sex"] for example in examples)
        print("Genders:", gender)

        output_format_err = 0
        for example in examples:
            if income_pred_label(example["output"]) is None:
                print("Detect OFE:", example["output"])
                output_format_err += 1

        # effectiveness metrics
        pos = sum(x["target"].startswith(">50K") for x in examples)
        neg = sum(x["target"].startswith("<=50K") for x in examples)
        true_pos = sum(income_pred_label(x["output"]) == 1 and x["target"].startswith(">50K") for x in examples)
        false_pos = sum(income_pred_label(x["output"]) == 1 and x["target"].startswith("<=50K") for x in examples)
        false_neg = sum(income_pred_label(x["output"]) == 0  and x["target"].startswith(">50K")for x in examples)
        true_neg = sum(income_pred_label(x["output"]) == 0 and x["target"].startswith("<=50K") for x in examples)

        print(f"total={total} pos={pos} neg={neg} TP={true_pos} FP={false_pos} FN={false_neg} TN={true_neg} OFE={output_format_err}")
        print(f"{total} {pos} {neg} {true_pos} {false_pos} {false_neg} {true_neg} {output_format_err}")
        print(f"Sum(TP, TN, FP, FN, OFE) == {output_format_err + true_pos + true_neg + false_pos + false_neg}")

    acc = (true_pos + true_neg) / total 
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * precision * recall / (precision + recall)

    return acc, precision, recall, f1
