import evaluate


def evaluate_summaries(predictions, references):
    """
    Evaluates generated summaries against reference summaries using ROUGE, BLEU and METEOR.

    Args:
        predictions (list): A list of generated summaries (strings).
        references (list): A list of reference summaries (strings).

    Returns:
        dict: A dictionary containing the ROUGE scores.
    """
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")

    rouge_results = rouge.compute(predictions=predictions, references=references)
    bleu_results = bleu.compute(predictions=predictions, references=references)
    meteor_results = meteor.compute(predictions=predictions, references=references)

    return {
        "rouge": rouge_results,
        "bleu": bleu_results,
        "meteor": meteor_results,
    }
