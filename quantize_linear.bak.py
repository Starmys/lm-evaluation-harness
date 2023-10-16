import torch

from smoothquant.llama import LlamaForCausalLM


class QuantizeLinear(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        w_bits: int,
        a_bits: int,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        w_scales: torch.Tensor,
        g_idx: torch.Tensor,
        bias: torch.Tensor = None,
        smooth: torch.Tensor = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.w_maxq = 2 ** self.w_bits - 1
        self.a_maxq = 2 ** self.a_bits - 1
        self.group_size = in_features // qzeros.shape[0]
        self.qweight = torch.nn.Parameter(
            torch.zeros((in_features, out_features), dtype=torch.int32, device=qweight.device),
            requires_grad=False,
        )
        self.qzeros = torch.nn.Parameter(
            torch.zeros((in_features // self.group_size, out_features), dtype=torch.int32, device=qzeros.device),
            requires_grad=False,
        )
        self.g_idx = torch.nn.Parameter(g_idx, requires_grad=False)
        self.w_scales = torch.nn.Parameter(w_scales, requires_grad=False)
        if smooth == 'fake':
            self.fake_smooth = True
            self.smooth = None
        else:
            self.fake_smooth = False
            self.smooth = smooth
        self.bias = bias
        n = 32 // self.w_bits
        for j, base in enumerate(range(0, 32, self.w_bits)):
            self.qweight[j::n, :] = qweight.bitwise_right_shift(base).bitwise_and(self.w_maxq)
        for j, base in enumerate(range(0, 32, self.w_bits)):
            self.qzeros[:, j::n] = qzeros.bitwise_right_shift(base).bitwise_and(self.w_maxq) + 1

    def forward(self, x: torch.Tensor):
        weight = (self.qweight - self.qzeros[self.g_idx]).to(x.dtype) * self.w_scales[self.g_idx]
        if self.fake_smooth:
            max_input = torch.max(torch.abs(x), dim=-2).values.detach().mean(0)
            max_weight = torch.max(torch.abs(weight), dim=-1).values.detach()
            scale = torch.sqrt(max_input * max_weight)
            x = x * (scale / (max_input + 1e-6))
            real_weight = weight * (scale / (max_weight + 1e-6)).unsqueeze(1)
        else:
            real_weight = weight
        if self.a_bits < x.element_size() * 8:
            if self.smooth is not None:
                x /= self.scale
            act_scales = torch.max(torch.abs(x), dim=-1, keepdim=True).values
            s = (2 ** (self.a_bits - 1) - 1) / (act_scales + 1e-6)
            x = torch.round(x * s).div(s + 1e-6)
        if self.bias is None:
            return torch.matmul(x, real_weight)
        else:
            return torch.matmul(x, real_weight) + self.bias


def get_quant_linear(
    linear: torch.nn.Linear,
    name: str,
    quant_params: dict[str, torch.Tensor],
    w_bits: int,
    a_bits: int,
    smooth_params: dict[str, torch.nn.Linear] = None,
):
    if smooth_params is None:
        smooth = None
    elif smooth_params == 'fake':
        smooth = 'fake'
    else:
        smooth = smooth_params[f'{name}.smooth']
    return QuantizeLinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        w_bits=w_bits,
        a_bits=a_bits,
        qweight=quant_params[f'{name}.qweight'],
        qzeros=quant_params[f'{name}.qzeros'],
        w_scales=quant_params[f'{name}.scales'],
        g_idx=quant_params[f'{name}.g_idx'],
        bias=linear.bias,
        smooth=smooth,
    )


def load_quant(
    model: LlamaForCausalLM,
    quant_checkpoint: str,
    w_bits: int,
    a_bits: int,
    smooth_checkpoint: str = None,
):
    model = model.cuda()
    quant_params = torch.load(quant_checkpoint)
    quant_params = {k: v.cuda() for k, v in quant_params.items()}
    if smooth_checkpoint is None:
        smooth_params = None
    elif smooth_checkpoint == 'fake':
        smooth_params = 'fake'
    else:
        smooth_params = torch.load(smooth_checkpoint)
        smooth_params = {k: v.cuda() for k, v in smooth_params.items()}
    print('Loading quant params:')
    for i, layer in enumerate(model.model.layers):
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            setattr(
                layer.self_attn,
                name,
                get_quant_linear(
                    getattr(layer.self_attn, name),
                    f'model.layers.{i}.self_attn.{name}',
                    quant_params,
                    w_bits,
                    a_bits,
                    smooth_params,
                ),
            )
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            setattr(
                layer.mlp,
                name,
                get_quant_linear(
                    getattr(layer.mlp, name),
                    f'model.layers.{i}.mlp.{name}',
                    quant_params,
                    w_bits,
                    a_bits,
                    smooth_params,
                ),
            )
        print(f'\tLayer #{i}')
    print('Done.')
    return model


if __name__ == '__main__':
    linear = torch.nn.Linear(in_features=4096, out_features=4096)
    quant_params = torch.load('/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llama7b-4bit.pt')
    quant_linear = get_quant_linear(linear, 'model.layers.0.self_attn.q_proj', quant_params, 4, 32).cuda()
    import ipdb; ipdb.set_trace()
