# Scripts

Examples of scripts to run the codes.

## Some Arguments

| Arguments      | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| knowledge_mode | Which information to use. Choose from RAW, CS, CSA, and CSF. |
| position_mode  | Which position embedding method to use. Choose from vanilla, sin, trainable, and relative. |
| specific       | Which dataset to use. Choose from iemocap, dailydialog, emory, and meld. |
| window_size    | Using how many utterances above and below.                   |
| use_crf        | Whether to use a CRF layer.                                  |

For knowledge_mode, RAW represents using RoBERTa without any contexts, CS represents using both historical contexts and historical speaker-specific contexts, CSA represents using historical contexts, historical speaker-specific contexts, and pseudo future contexts, CSF represents using historical contexts, historical speaker-specific contexts, and real future contexts.

More arguments can refer to `../arguments_spa.py`
