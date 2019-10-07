import time

from pathlib import Path
from itertools import islice

import torch
import tqdm

import fairseq
from fairseq.models.roberta import RobertaModel

from sotabencheval.natural_language_inference import MultiNLI

label_map = {
    0: 'contradiction',
    1: 'neutral',
    2: 'entailment'
}
def predict_mnli(hub_model, tokens):
    hub_model.eval()
    result = hub_model.predict('mnli', tokens).argmax().item()
    return label_map[result]

def evaluate_roberta():
    evaluator = MultiNLI(
        model_name="ROBERTa",
        paper_arxiv_id="1907.11692",
        paper_pwc_id="roberta-a-robustly-optimized-bert-pretraining",
        local_root=Path.home()/"sotabench/data/multinli",
        #expected accuracy on devset: 90.2/90.2
    )
    # we can either read the data our selfs from files in path = evaluator.dataset_paths
    
    hub_model = RobertaModel.from_pretrained(
       "roberta.large.mnli", data_name_or_path='.')
    hub_model = hub_model.to("cuda")
    
    start = time.time()
    # TODO: figure out how ROBERTa was handling longer sentences
    # TODO: add batching
    def predict_generator(sentences_stream):
        for pairId, sentences in tqdm.tqdm(list(sorted(sentences_stream, key=lambda x: -(len(x[1][0])+len(x[1][1]))))):
            tokens = hub_model.encode(*sentences)
            strip = max(0, len(tokens) - 512) // 2
            tokens = tokens[strip:-strip-2]
            pred_class = predict_mnli(hub_model, tokens)
            
            # s0 = hub_model.encode(*sentences)
            # s1 = hub_model.encode(sentences[1])
            # sep = hub_model.task.source_dictionary.encode_line('<s>')[:1].long()
            # tokens = torch.cat([s0[:256], sep, s1[:255]], 0)
            # pred_class = predict_mnli(hub_model, tokens)
            yield pairId, pred_class

    data = evaluator.dataset
    #data = islice(data, 100)
    evaluator.eval(predict_generator(data))
    print(evaluator.results)
    end = time.time()
    print("Evaluation time", end - start, " sec")

def main():
    evaluate_roberta()
#%%
if __name__ == '__main__':
    main()
