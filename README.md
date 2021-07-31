# ColBERT_FindZebra
 
This Git repository was made and used for writing a thesis. In short, the two models "ColBERT_MC" and "ColBERT_MC_retr_train" are modified versions of the original ColBERT model: https://github.com/stanford-futuredata/ColBERT, modified to be used to solve a multiple choice test with 5 options. A big thanks to the authors for their great model.

## Thesis specifics

- Model1 to Model8 have all been trained using the original ColBERT implementation: https://github.com/stanford-futuredata/ColBERT.
- Model9 and Model10 have both been trained using the "ColBERT_MC" model.
- Model11 has been trained using the "ColBERT_MC_retr_train" model.

All of the models have been evaluated using the testing procedure found in "ColBERT_MC". The results will be saved as csv files (one for each GPU), which contains the GPU rank, an unique index (unique across all GPUs), the retrieval rank, the predicted option ([1,2,3,4,5]), and the cosine similarity scores given to each option.

Technically, ColBERT_MC_retr_train should be able to do everything ColBERT_MC can, but I have not formally tested it. The only difference between the two models, is that ColBERT_MC_retr_train keeps track of how far through the training dataset it is, and stops training when there is too little data left for all of the GPUs to run one more training iteration. And then it saves a final checkpoint. This change was needed because of the small data size of the fine-tuning training triples, i.e., the model would always use the whole training set for training. If some of the GPUs continue training after some of them have stopped, they get stuck in the training waiting for the now idle GPUs. There should not be any difference when training on only one GPU. FYI: ColBERT_MC_retr_train needs an extra input "iteration". If only training for one iteration, just pass it as 0.

To run the models, simply follow the same procedure as for the original ColBERT model. However, the triples need to include 5 extra columns: one column per query option, where the first column contains the correct answer. The latter is only important for training, but also makes evaluating the models easier.

Training example:\
CUDA_VISIBLE_DEVICES= CUDA_DEVICES   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # e.g. 0,1,2,3 \
python -m torch.distributed.launch \
--nproc_per_node= NPROC               &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# e.g. 4 \
-m colbert.train \
--root /path/to/experiment \
--amp \
--accum 1 \
--similarity cosine \
--query_maxlen 250                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Can be changed, but fits to the MedQA dataset \
--doc_maxlen 75                       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Can be changed, but fits to the MedQA dataset \
--mask-punctuation \
--run experiment.psg.cosine \
--bsize batch_size                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Should be divisible with the number of GPUs \
--checkpoint checkpoint.dnn           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# If the model should start from existing checkpoint \
--triples mc_triples.tsv              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# The triples. For multiple choice, these triples should contain the 5 options as 5 new columns, with the correct answer as first column \
--experiment experiment-psg           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# The name of the experiment \
--iteration {iteration}               &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Only needed for ColBERT_MC_retr_train \

### Data, checkpoints, and scrips

Because of a very bad internet connection, the only way I could think of to upload all my data, was via OneDrive. The data is available to anyone with a DTU account.

- FindZebra, MedQA, and UMLS: https://dtudk-my.sharepoint.com/:f:/g/personal/s190619_dtu_dk/EilueYgitzZEofbBLITYpN4BjLENvba5Ib6kOCI79MCbcw?e=jKOgLH
- MS MARCO filtered to only contain medical domain data: https://dtudk-my.sharepoint.com/:f:/g/personal/s190619_dtu_dk/ElEhh-tluwBDvVwspCSiSVEB31F1vBbp88zmwkHUsw8MLw?e=4ijXTG
- Checkpoints: https://dtudk-my.sharepoint.com/:f:/g/personal/s190619_dtu_dk/Emi9rLf6IepMijWpaXydUkQB1Xsullw3NNN8QriysCbz_A?e=ohnPtd
- Retrievals: https://dtudk-my.sharepoint.com/:f:/g/personal/s190619_dtu_dk/ElUdGRC8PMRCpKzRDfFVk_EBKQ5tRCxTsgpP24aZ8f1kKg?e=bI8DTQ
- Results: https://dtudk-my.sharepoint.com/:f:/g/personal/s190619_dtu_dk/Ek_zu9AEDqVPhKekBMGotZMB2vVmzIcUEwVrpGhyTI6SEQ?e=oQR0d9
- Data transformation scripts: https://dtudk-my.sharepoint.com/:f:/g/personal/s190619_dtu_dk/EkAWJATaTFZDiCZOAdpCfhQB0vsWMzznGpjdACPRGzETgA?e=lbiMeq
- Scripts and wrappers for running models: https://dtudk-my.sharepoint.com/:f:/g/personal/s190619_dtu_dk/EmoF0Hr1AKNFp9BhC8WgshgBO5GxVSy070cXX5WvB5nMCQ?e=9UtxaN

I am afraid there is quite a bit of hard-coding in the scripts and wrappers used for running the models, since it was the easiest to work with in the context of my thesis.


