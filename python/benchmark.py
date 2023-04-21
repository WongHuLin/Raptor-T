from benchmark.async_ import async_
from benchmark.attention_test import attention_test
from benchmark.CTAs_num_test import CTAs_num_test
from benchmark.end2end import end2end
from benchmark.mem_bound_op import kernel_fusion_mem_op
import argparse


functions = {"async_":async_,
             "attention_test":attention_test,
             "CTAs_num_test":CTAs_num_test,
             "end2end":end2end,
             "kernel_fusion_mem_op":kernel_fusion_mem_op}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size',
                        help='Batch size to use.',
                        default=4, type=int)
    parser.add_argument('-s', '--sequence-length',
                        help='The average sequence length to use. Defaults to 2048',
                        default=3840, type=int)
    parser.add_argument('-m', '--model-name', required=False,
                        default='pytorch',
                        help='model name to launch')
    parser.add_argument('-t', '--thread-block',
                        default=0,type=int,
                        help='raptor-t FMHA CTA number')
    parser.add_argument('-ba', '--balanced',
                        default=False,type=bool,
                        help='balanced attention')
    parser.add_argument('-a', '--async-',
                        default=False,type=bool,
                        help='async processing')
    parser.add_argument('-k', '--kernel-fusion',
                        default=False,type=bool,
                        help='kernel fusion option')
    parser.add_argument('-be', '--benchmark',
                        default="end2end",type=str,
                        help='benchmark')
    args, _ = parser.parse_known_args()
    args.bigbird_dir = "/workspace/models/bigbird"
    args.longformer = "/workspace/models/longformer"
    args.ft_longformer_lib = "/workspace/FastTransformer/build/lib/libth_transformer.so"
    f = functions[args.benchmark]
    f(args)


if __name__ == '__main__':
    main()

