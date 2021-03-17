from models import dcgan_eqlr, dusty


def define_G(cfg):
    masker_type, backbone_type = cfg.arch.split("/")

    if backbone_type.lower() == "dcgan_eqlr":
        G = dcgan_eqlr.Generator(
            in_ch=cfg.in_ch,
            out_ch=dict(cfg.out_ch),
            ch_base=cfg.ch_base,
            ch_max=cfg.ch_max,
            shape=cfg.shape,
        )
    else:
        raise NotImplementedError

    if masker_type == "dusty1":
        G = dusty.DUSty1(G, tau=cfg.tau, drop_const=cfg.drop_const)
    elif masker_type == "dusty2":
        G = dusty.DUSty2(G, tau=cfg.tau, drop_const=cfg.drop_const)
    elif masker_type == "none":
        pass
    else:
        raise NotImplementedError
    return G


def define_D(cfg):
    if cfg.arch.lower() == "dcgan_eqlr":
        D = dcgan_eqlr.Discriminator(
            in_ch=cfg.in_ch,
            ch_base=cfg.ch_base,
            ch_max=cfg.ch_max,
            shape=cfg.shape,
        )
    else:
        raise NotImplementedError
    return D
