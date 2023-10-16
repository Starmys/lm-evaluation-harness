import torch

from transformers import LlamaForCausalLM


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

        group_size = in_features // qzeros.shape[0]
        qweight_expand = torch.zeros((in_features, out_features), dtype=torch.int32, device=qweight.device)
        qzeros_expand = torch.zeros((in_features // group_size, out_features), dtype=torch.int32, device=qzeros.device)

        if smooth == 'fake':
            self.fake_smooth = True
            self.smooth = None
        else:
            self.fake_smooth = False
            self.smooth = smooth
        n = 32 // self.w_bits
        for j, base in enumerate(range(0, 32, self.w_bits)):
            qweight_expand[j::n, :] = qweight.bitwise_right_shift(base).bitwise_and(self.w_maxq)
        for j, base in enumerate(range(0, 32, self.w_bits)):
            qzeros_expand[:, j::n] = qzeros.bitwise_right_shift(base).bitwise_and(self.w_maxq) + 1

        self.weight = torch.nn.Parameter(
            ((qweight_expand - qzeros_expand[g_idx]) * w_scales[g_idx]).to(torch.float16),
            requires_grad=False,
        )
        self.bias = bias

    def forward(self, x: torch.Tensor):
        if self.fake_smooth:
            max_input = torch.max(torch.abs(x), dim=-2).values.detach().mean(0)
            max_weight = torch.max(torch.abs(self.weight), dim=-1).values.detach()
            scale = torch.sqrt(max_input * max_weight)
            x = x * (scale / (max_input + 1e-6))
            real_weight = self.weight * (scale / (max_weight + 1e-6)).unsqueeze(1)
        else:
            real_weight = self.weight
        if self.a_bits < x.element_size() * 8:
            if self.smooth is not None:
                x /= self.smooth
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
    ignore_components: list = None,
):
    if ignore_components is not None:
        for component in ignore_components:
            if component in name:
                print(f'[ignored] {name}', end=' ')
                return linear
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


def quant_self_attn_llama(
    layer: torch.nn.Module,
    idx: int,
    quant_params: dict[str, torch.Tensor],
    w_bits: int,
    a_bits: int,
    smooth_params: dict[str, torch.nn.Linear] = None,
    prefix: str = '',
    path: str = 'model.layers',
    ignore_components: list = None,
):
    for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        setattr(
            layer.self_attn,
            name,
            get_quant_linear(
                getattr(layer.self_attn, name),
                f'{path}.{idx}.{prefix}self_attn.{name}',
                quant_params,
                w_bits,
                a_bits,
                smooth_params,
                ignore_components,
            ),
        )
    for name in ['gate_proj', 'up_proj', 'down_proj']:
        setattr(
            layer.mlp,
            name,
            get_quant_linear(
                getattr(layer.mlp, name),
                f'model.layers.{idx}.{prefix}mlp.{name}',
                quant_params,
                w_bits,
                a_bits,
                smooth_params,
                ignore_components,
            ),
        )


def quant_self_attn_mpt(
    layer: torch.nn.Module,
    idx: int,
    quant_params: dict[str, torch.Tensor],
    w_bits: int,
    a_bits: int,
    smooth_params: dict[str, torch.nn.Linear] = None,
    prefix: str = '',
    path: str = 'transformer.blocks',
    ignore_components: list = None,
):
    for name in ['Wqkv', 'out_proj']:
        setattr(
            layer.attn,
            name,
            get_quant_linear(
                getattr(layer.attn, name),
                f'{path}.{idx}.{prefix}attn.{name}',
                quant_params,
                w_bits,
                a_bits,
                smooth_params,
                ignore_components,
            ),
        )
    for name in ['up_proj', 'down_proj']:
        setattr(
            layer.ffn,
            name,
            get_quant_linear(
                getattr(layer.ffn, name),
                f'{path}.{idx}.{prefix}ffn.{name}',
                quant_params,
                w_bits,
                a_bits,
                smooth_params,
                ignore_components,
            ),
        )


def quant_cross_attn(
    layer: torch.nn.Module,
    idx: int,
    quant_params: dict[str, torch.Tensor],
    w_bits: int,
    a_bits: int,
    smooth_params: dict[str, torch.nn.Linear] = None,
    prefix: str = '',
    path: str = 'model.layers',
    ignore_components: list = None,
):
    for name in ['to_q', 'to_kv', 'to_out']:
        setattr(
            layer.attn,
            name,
            get_quant_linear(
                getattr(layer.attn, name),
                f'{path}.{idx}.{prefix}attn.{name}',
                quant_params,
                w_bits,
                a_bits,
                smooth_params,
                ignore_components,
            ),
        )
    for pos in [1, 3]:
        layer.ff[pos] = get_quant_linear(
            layer.ff[pos],
            f'{path}.{idx}.{prefix}ff.{pos}',
            quant_params,
            w_bits,
            a_bits,
            smooth_params,
            ignore_components,
        )


def load_quant(
    model: LlamaForCausalLM,
    quant_checkpoint: str,
    w_bits: int,
    a_bits: int,
    smooth_checkpoint: str = None,
    ignore_layers: list = None,
    ignore_components: list = None,
):
    w_bits = int(w_bits)
    a_bits = int(a_bits)
    if type(ignore_layers) is str:
        ignore_layers = [int(l) for l in ignore_layers.split(' ')]
    if type(ignore_components) is str:
        ignore_components = ignore_components.split(' ')
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
    quant_args = {
        'quant_params': quant_params,
        'w_bits': w_bits,
        'a_bits': a_bits,
        'smooth_params': smooth_params,
        # 'ignore_components': ignore_components,
    }
    print('Loading quant params:')
    try:
        path = 'model.layers'
        layers = model.model.layers
        quant_self_attn = quant_self_attn_llama
    except AttributeError:
        path = 'transformer.blocks'
        layers = model.transformer.blocks
        quant_self_attn = quant_self_attn_mpt
    for i, layer in enumerate(layers):
        quant_args['idx'] = i
        print(f'\tLayer #{i}', end=' ')
        if ignore_layers is not None and i in ignore_layers:
            # print('[ignored]')
            # continue
            ignore_args = {'ignore_components': ignore_components}
        else:
            ignore_args = {}
        if hasattr(layer, 'decoder_layer'):
            if hasattr(layer, 'gated_cross_attn_layer') and layer.gated_cross_attn_layer is not None:
                # try:
                quant_cross_attn(**quant_args, **ignore_args, layer=layer.gated_cross_attn_layer, prefix='gated_cross_attn_layer.', path=path)
                # except KeyError:
                #     print(f'\tWarning: quant Flamingo layer without cross attention')
            else:
                print(f'[warning] cross attention block not found', end=' ')
            quant_self_attn(**quant_args, **ignore_args, layer=layer.decoder_layer, prefix='decoder_layer.', path=path)
        else:
            quant_self_attn(**quant_args, **ignore_args, layer=layer, path=path)
        print(f'')
    print('Done.')
    return model


if __name__ == '__main__':
    linear = torch.nn.Linear(in_features=4096, out_features=4096)
    quant_params = torch.load('/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llama7b-4bit.pt')
    quant_linear = get_quant_linear(linear, 'model.layers.0.self_attn.q_proj', quant_params, 4, 32).cuda()
    import ipdb; ipdb.set_trace()
