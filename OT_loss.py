# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch

@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("params.optimization.sentence_avg")


@register_criterion("Waserstein_loss", dataclass=CrossEntropyCriterionConfig)
class WassersteinLoss(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--parse-penalty', default=1.0, type=float,
                            help='penalty of parsing loss')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        print(loss)
        # return None
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        origin_target = model.get_targets(sample, net_output)
        # print(net_output[0].shape)
        # target_embedding = model.get_decoder_embedding(origin_target)
        # # print(target_embedding)
        # print(target_embedding.shape)
        # print(lprobs.shape)
        # lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)
        
        # target = origin_target.cpu().numpy()
        # np_log_probs = log_probs.cpu().detach().numpy()
        
        batch_size, seq_len, num_vocab = lprobs.shape
        np_labels = torch.zeros_like(lprobs)
        base_epsi = 0.4
        smoothed = base_epsi/num_vocab
        np_labels+=smoothed
        # print(target[0][0])
        for i in range(batch_size):
            for j in range(seq_len):
                np_labels[i][j][target[i][j]] += (1-base_epsi)


        # return None
        # loss = F.nll_loss(
        #     lprobs,
        #     target,
        #     ignore_index=self.padding_idx,
        #     reduction="sum" if reduce else "none",
        # )
        loss = torch.norm(np_labels-lprobs, p='fro')
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
