from pathlib import Path
from typing import Optional, Sequence, TypeVar, override

from tap import Tap

from utils.base_argument_parser import BaseArgumentParser
from utils.enums import DATASET_TYPE
from utils.metrics import MetricFunction

TapType = TypeVar("TapType", bound="Tap")

class EvalArgumentParser(BaseArgumentParser):
    model_dir: str                            # Path to the model directory
    test_set_paths: list[Path]=None           # List of paths to test csv files
    test_cache_paths: list[Path] = [] # Paths to cache files. If set, the cache files are used instead of the test files
    test_dataset_types: list[DATASET_TYPE]   # Dataset type of the test set
    test_set_metrics: list[MetricFunction] = [] # Metrics to use for test
    save_dir: Optional[Path] = None            # Directory to save the experiment artifacts
    save_testset_predictions: list[bool] = [] # If true, save the predictions for the test set
    additional_metrics: Optional[list[MetricFunction]] = None # Additional metrics to use for validation

    _config_name: str = "eval_config"       # Name of the configuration object in config files

    @override
    def verify_arguments(self):
        super().verify_arguments()
        assert len(self.test_set_paths) == len(self.test_dataset_types), f"Number of test set paths ({len(self.test_set_paths)}) must be equal to the number of test dataset types ({len(self.test_dataset_types)})"
        
        match self.model:
            case "stem" | "rope_stem" | "modded_nanoGPT_stem" | "modded_nanoGPT":
                for test_dataset_type in self.test_dataset_types:
                    assert test_dataset_type not in ["pretokenized", "efficient_prompt"], f"Test dataset type {test_dataset_type} is not supported for {self.model}. Use number datasets instead."
            case "gpt2" | "rope_gpt2":
                for test_dataset_type in self.test_dataset_types:
                    assert test_dataset_type not in ["number", "efficient_number_prompt"], f"Test dataset type {test_dataset_type} is not supported for {self.model}. Use prompt datasets instead."
        
        if self.test_set_metrics == []:
            self.test_set_metrics = [MetricFunction(MetricFunction.TOKEN_EQUALITY)] * len(self.test_set_paths)
        else:
            assert len(self.test_set_metrics) == len(self.test_set_paths), f"Number of validation set metrics ({len(self.test_set_metrics)}) must be equal to the number of validation sets ({len(self.test_set_paths)}"
        
        if self.save_testset_predictions == []:
            self.save_testset_predictions = [True] * len(self.test_set_paths)
        else:
            assert len(self.save_testset_predictions) == len(self.test_set_paths), f"Number of save test set predictions ({len(self.save_testset_predictions)}) must be equal to the number of test sets ({len(self.test_set_paths)})"
            for i, save_testset_prediction in enumerate(self.save_testset_predictions):
                assert not save_testset_prediction or self.test_dataset_types[i] in ["efficient_number_prompt", "efficient_prompt", "efficient_number_prompt_pos"], f"Test dataset type {self.test_dataset_types[i]} is not supported for saving test set predictions. Use 'efficient_number_prompt' or 'efficient_prompt' instead."
        
        if self.test_cache_paths == []:
            self.test_cache_paths = [None] * len(self.test_set_paths)
        return self
    
    @override
    def parse_args(self: TapType, args: Optional[Sequence[str]] = None, known_only: bool = False, legacy_config_parsing=False) -> "EvalArgumentParser": # pyright: ignore[reportInvalidTypeVarUse]
        return super().parse_args(args, known_only=known_only, legacy_config_parsing=legacy_config_parsing)
