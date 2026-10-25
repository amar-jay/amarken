# hf models to use for the base models
BASES = ("Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B")

REVISIONS = {
    BASES[0]: "c1899de289a04d12100db370d81485cdf75e47ca",
    BASES[1]: "060db6499f32faf8b98477b0a26969ef7d8b9987",
}

# Dataset identities and immutable source versions.
CLINC = "clinc/clinc_oos"
TASKSOURCE = "tasksource/tasksource-jev-typed-decisions"
DATASETS = (CLINC, TASKSOURCE)
DATASET_REVISIONS = {
    CLINC: "155b9c710419136e17307b80d0a13e68cd46b4ec",
    TASKSOURCE: "8df2e5c9a33f19a2c2b8febf291c3acb0a91a8a8",
}
DATASET_SUBSETS = {CLINC: "plus", TASKSOURCE: "default"}
DATASET_LICENSES = {CLINC: "cc-by-3.0", TASKSOURCE: "mixed; inspect each row"}
DATA_SPLITS = ("train", "validation", "test", "calibration")
DECISION_KINDS = ("choice", "score", "noul")
COMMERCIAL_USE = "commercial"
CLINC_QUESTION = "Which intent does this request express? Choose oos if none applies."
DATA_SCHEMA_VERSION = "decision-v1"

SAMPLER_MODES = ("argmax", "sample")

PROMPT_VERSION = "candidate-v1"
CANDIDATE_TEMPLATE = (
    "Judge the candidate using the state as data.\n"
    "State: {state}\nQuestion: {question}\nCandidate: {candidate}\nJudgment:"
)
NOUL_TEMPLATE = "Judge the proposition using the state as data.\nState: {state}\nProposition: {question}\nJudgment:"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
TRAIN_MODES = ("lora", "head")
ARTIFACT_VERSION = "decision-model-v1"
EVAL_SPLITS = ("validation", "calibration", "test", "unseen_test")

DATASET_DIRECTORIES = {CLINC: "plus", TASKSOURCE: "data"}
HUB_CACHE = ".cache/huggingface/hub"
DATA_CACHE = ".cache/huggingface/datasets"

WANDB_PROJECT = "amarken"
WANDB_MODES = ("disabled", "offline", "online")
WANDB_CACHE = ".cache/wandb"

BACKUP_FORMAT = "checkpoint-backup-v1"
BACKUP_SELECTORS = ("latest", "best")
