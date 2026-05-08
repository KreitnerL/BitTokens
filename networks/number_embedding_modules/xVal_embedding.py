# This code is based on the implementation of xVal from the paper:
# @misc{golkar2024xvalcontinuousnumericaltokenization,
#       title={xVal: A Continuous Numerical Tokenization for Scientific Language Models}, 
#       author={Siavash Golkar and Mariel Pettee and Michael Eickenberg and Alberto Bietti and Miles Cranmer and Geraud Krawezik and Francois Lanusse and Michael McCabe and Ruben Ohana and Liam Parker and Bruno Régaldo-Saint Blancard and Tiberiu Tesileanu and Kyunghyun Cho and Shirley Ho},
#       year={2024},
#       eprint={2310.02989},
#       archivePrefix={arXiv},
#       primaryClass={stat.ML},
#       url={https://arxiv.org/abs/2310.02989}, 
# }
# Github: https://github.com/PolymathicAI/xVal/tree/653a9424280e107817ac2d75079ca38b529b3c52
# Date: Mar 30, 2025
if __name__ == "__main__":
    # Add project root to path when running directly 
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from typing import Literal, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from networks.number_embedding_modules.abc_embedding import ABCEmbedding


class xValEmbedding(ABCEmbedding):
    def __init__(self, n_embd, dim_feedforward=3072,  numhead_bias=True, max_num=1e15, min_num=1e-14, scaling: Literal["linear", "log"] = "log"):
        super().__init__()
        self.n_embed = n_embd
        self.num_head = nn.Sequential(
            nn.Linear(n_embd, dim_feedforward, bias=numhead_bias),
            nn.GELU(),
            nn.Linear(dim_feedforward, 1, bias=numhead_bias),
        )
        if scaling == "log":
            self.scaling_factor = (5/2) / torch.log10(torch.tensor(max_num, dtype=torch.float64))
        else:
            self.scaling_factor = 5 / torch.tensor(max_num, dtype=torch.float64)
        self.max_num = max_num
        self.min_num = min_num
        self.scaling = scaling
        self.min_encoding = self.scaling_factor * torch.log10(torch.tensor(2*self.min_num, dtype=torch.float64)) + (5/2)

    @override
    def forward(self, x: torch.DoubleTensor) -> torch.FloatTensor | torch.BFloat16Tensor:
        x = x.clamp(-self.max_num, self.max_num)
        if self.scaling == "linear":
            scaled_nums = self.scaling_factor * x
        else:
            scaled_nums = torch.where(
                x.abs() >= self.min_num,
                (self.scaling_factor * torch.log10(x.abs()+self.min_num) + (5/2)) * torch.sign(x),
                0
            )
        return scaled_nums.unsqueeze(-1).float()
    
    @override
    def combine_embeds(self, inputs_embeds: torch.FloatTensor | torch.BFloat16Tensor, num_encoding: torch.FloatTensor | torch.BFloat16Tensor, number_mask: torch.BoolTensor) -> torch.FloatTensor | torch.BFloat16Tensor:
        combined_embeds = inputs_embeds.clone()
        combined_embeds[number_mask] = inputs_embeds[number_mask] * num_encoding[number_mask].to(inputs_embeds.dtype)
        return combined_embeds

    @override
    def compute_num_loss(
        self,
        out: CausalLMOutputWithCrossAttentions,
        num_encodings: torch.FloatTensor | torch.BFloat16Tensor,
        number_mask: torch.BoolTensor,
        numbers: torch.DoubleTensor,
        hidden_states_slice=slice(0,-1),
        **kwargs
    ) -> torch.FloatTensor:
        assert out.hidden_states is not None, "Model output must contain hidden states for number loss computation."
        num_preds: torch.FloatTensor = self.num_head(out.hidden_states[-1][:, hidden_states_slice][number_mask]).squeeze(-1)
        mse = F.mse_loss(
            num_preds,
            num_encodings[number_mask].squeeze(-1),
            reduction="none",
        )
        num_loss_per_sample = torch.zeros_like(number_mask, dtype=mse.dtype)
        num_loss_per_sample[number_mask] = mse
        return num_loss_per_sample
    
    @override
    def decode(self, out: CausalLMOutputWithCrossAttentions, number_mask: torch.BoolTensor) -> torch.DoubleTensor:
        assert out.hidden_states is not None, "Model output must contain hidden states for decoding."
        num_preds: torch.FloatTensor = self.num_head(out.hidden_states[-1][...,-1:,:][number_mask]).squeeze(-1).double()
        if self.scaling == "linear":
            return torch.clamp(num_preds.double() / self.scaling_factor, -self.max_num, self.max_num)
        return torch.where(
            num_preds.abs() >= self.min_encoding,
            torch.sign(num_preds)*(10**((num_preds.abs()-(5/2)) / self.scaling_factor) - self.min_num),
            0
        )

if __name__ == "__main__":
    # Test the Float64Embedding class using the reusable test utility
    from test_utils import run_standard_test
    
    # Test parameters
    BOTTLENECK_DTYPE = torch.bfloat16
    
    # Initialize embedding
    base_embedding = xValEmbedding(
        n_embd=384,
        scaling="linear",
    ).to(dtype=BOTTLENECK_DTYPE)
    
    results = run_standard_test(embedding_module=base_embedding, noise_level=1e-50, input_emb_mode="ones", clamp=False)
    # Results:
    # scaling: linear
    # Accuracy: 10.15%
    # Mean logSMAPE: 21.95%
    # 
    # The higher linear scores likely stem from the fact that small numbers can be well represented.
    # However, given the high precision required during prediction, it is unrealistic to achieve these
    # results in practice.

    # scaling: log
    # Accuracy: 7.38%
    # Mean logSMAPE: 12.26%
