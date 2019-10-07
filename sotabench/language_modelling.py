#%%
import torch
import numpy as np

import fairseq
import hubconf
from fairseq.models.transformer_lm import TransformerLanguageModel
from fairseq.sequence_scorer import SequenceScorer
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.data import LMContextWindowDataset
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from sotabencheval.language_modelling import WikiText103Evaluator, WikiText2Evaluator

from fairseq.models.transformer_lm import TransformerLanguageModel

#%%
def evaluate_language_model(evaluator, model, parsed_args):
    """
    Adpted from eval_lm.main()
    """
    assert parsed_args.path is not None, '--path required for evaluation!'

    utils.import_user_module(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args.cpu

    task = tasks.setup_task(parsed_args)

    # Load ensemble
    print('| loading model(s) from {}'.format(parsed_args.path))
    models, args = model.models, model.args

    for arg in vars(parsed_args).keys():
        if arg not in {
            'self_target', 'future_target', 'past_target', 'tokens_per_sample',
            'output_size_dictionary', 'add_bos_token',
        }:
            setattr(args, arg, getattr(parsed_args, arg))

    # reduce tokens per sample by the required context window size
    args.tokens_per_sample -= args.context_window
    task = tasks.setup_task(args)

    # Load dataset splits
    task.load_dataset(args.gen_subset)
    dataset = task.dataset(args.gen_subset)
    if args.context_window > 0:
        dataset = LMContextWindowDataset(
            dataset=dataset,
            tokens_per_sample=args.tokens_per_sample,
            context_window=args.context_window,
            pad_idx=task.source_dictionary.pad(),
        )
    print('| {} {} {} examples'.format(
        args.data, args.gen_subset, len(dataset)))

    # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
    for model in models:
        model.make_generation_fast_()
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    assert len(models) > 0

    print('num. model params: {}'.format(sum(p.numel()
                                             for p in models[0].parameters())))

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens or 36000,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(*[
            model.max_positions() for model in models
        ]),
        ignore_invalid_inputs=True,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(task.target_dictionary, args.softmax_batch)

    score_sum = 0.
    count = 0

    if args.remove_bpe is not None:
        if args.remove_bpe == 'sentencepiece':
            raise NotImplementedError
        else:
            bpe_cont = args.remove_bpe.rstrip()
            bpe_toks = set(
                i
                for i in range(len(task.source_dictionary))
                if task.source_dictionary[i].endswith(bpe_cont)
            )
        bpe_len = len(bpe_cont)
    else:
        bpe_toks = None
        bpe_len = 0

    word_stats = dict()

    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        
        for sample in t:
            if 'net_input' not in sample:
                continue
           
            sample = utils.move_to_cuda(sample) if use_cuda else sample

            gen_timer.start()

            # compute scores for each model in the ensemble
            hypos = scorer.generate(models, sample)
            for hypo, *_ in hypos:
                evaluator.add(hypo['positional_scores'].float(), hypo['tokens'])
            if evaluator.cache_exists:
                break
            gen_timer.stop(sample['ntokens'])
    evaluator.save()
    print("evaluator.results: ", evaluator.results, evaluator)
    print('| Evaluated {} tokens in {:.1f}s ({:.2f} tokens/s)'.format(gen_timer.n,
                                                                      gen_timer.sum, 1. / gen_timer.avg))
    evaluator.print_stats()
    return evaluator.results

def evaluate_transformer_lm():
    evaluator = WikiText103Evaluator(
        model_name="Transformer (Adaptive inputs)",
        paper_arxiv_id="1809.10853",
        paper_pwc_id="adaptive-input-representations-for-neural",
        #expected perplexity: 18.70
    )
    hub_model = TransformerLanguageModel.from_pretrained("transformer_lm.wiki103.adaptive", data_name_or_path='./data-bin/')

    parser = options.get_eval_lm_parser()
    input_args = [
        hub_model.args.data,
        f"--path={hub_model.args.data}/../model.pt",
        "--sample-break-mode=complete",
        "--max-tokens=3072",
        "--context-window=2560",
        "--softmax-batch=1024",
        '--no-progress-bar'
    ]
    args = options.parse_args_and_arch(parser, input_args=input_args)

    evaluate_language_model(evaluator, hub_model, args)


def main():
    evaluate_transformer_lm()
#%%
if __name__ == '__main__':
    main()