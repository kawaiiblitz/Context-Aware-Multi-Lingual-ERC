import argparse
import torch
import transformers
from transformers import BertConfig, BertTokenizer, BertModel, \
                         RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForSequenceClassification,\
                         AlbertTokenizer, AlbertConfig, AlbertModel, \
                         BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from transformers.models.albert.modeling_albert import AlbertMLMHead

_GLOBAL_ARGS = None

_MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model': BertModel,
        'masked_lm': BertForMaskedLM,
        'head': BertOnlyMLMHead
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model': RobertaModel,
        'masked_lm': RobertaForMaskedLM,
        'head': RobertaLMHead,
        'classification': RobertaForSequenceClassification
    },
    'albert': {
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'model': AlbertModel,
        'masked_lm': AlbertForMaskedLM,
        'head': AlbertMLMHead
    },
    'roberta-bne': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model': RobertaModel,
        'masked_lm': RobertaForMaskedLM,
        'head': RobertaLMHead,
        'classification': RobertaForSequenceClassification
    }
}

def get_args_parser():
    parser = argparse.ArgumentParser(description="Command line interface for ERCMC.")
    
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data directory."
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or shortcut name of the model."
    )

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval.")

    parser.add_argument(
        "--use_crf", action="store_true", help="Whether to use CRF."
    )

    parser.add_argument(
        "--knowledge_mode", type=str, default='CSA', help="Which knowledge to use. Choose from RAW, CS, CSA, CSF."
    )

    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--evaluate_after_epoch",
        action="store_true",
        help="Whether to run evaluation after every epoch.",
    )
    
    parser.add_argument("--per_gpu_train_batch_size", default=2, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=2, type=int, help="Batch size per GPU/CPU for evaluation."
    )

    parser.add_argument("--position_mode", type=str, default='relative', help="Which position embedding method to use. Choose from vanilla, sin, trainable and relative.")
    parser.add_argument("--specific", type=str, default='iemocap', help="Which dataset to use. Choose from iemocap, dailydialog, emory and meld.")
    parser.add_argument("--window_size", default=5, type=int, help="Using how many utterances above and below.")
    parser.add_argument("--d_inner", default=3072, type=int, help="Hidden size of the position-wise feed forward.")
    parser.add_argument("--n_layers", default=1, type=int, help="Number of layers for multi-head self-attention.")
    parser.add_argument("--n_head", default=8, type=int, help="Number of heads for multi-head self-attention.")
    parser.add_argument("--n_position", default=200, type=int, help="Number of position for positon embedding. Only used fo sin and trainable.")
    parser.add_argument("--encoder_dropout", default=0.1, type=float, help="Dropout prob in encoder.")

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")

    parser.add_argument("--warmup_proportion", default=0, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=4.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=str, default='0.1', help="Log every X updates steps.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")

    parser.add_argument("--dropout_prob", type=float, default=0.1, help="Dropout rate.")
    
    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()

    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    return args

def get_args():
    return _GLOBAL_ARGS

def get_model_classes():
    return _MODEL_CLASSES
