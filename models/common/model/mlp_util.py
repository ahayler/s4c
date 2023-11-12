from .mlp import ImplicitNet
from .resnetfc import ResnetFC
from .debug_models import InterceptOnlyModel
from .segnet import SegNet


def make_mlp(conf, d_in, d_latent=0, allow_empty=False, **kwargs):
    mlp_type = conf.get("type", "mlp")  # mlp | resnet
    if mlp_type == "mlp":
        net = ImplicitNet.from_conf(conf, d_in + d_latent, **kwargs)
    elif mlp_type == "resnet":
        net = ResnetFC.from_conf(conf, d_in, d_latent=d_latent, **kwargs)
    elif mlp_type == "empty" and allow_empty:
        net = None
    else:
        raise NotImplementedError("Unsupported MLP type")
    return net


def make_intercept_model(d_in, d_out, **kwargs):
    return InterceptOnlyModel(d_in, d_out)

def make_segnet(d_in, d_out, d_hidden_list):
    return SegNet(d_in=d_in, d_out=d_out, d_hidden_list=d_hidden_list)



