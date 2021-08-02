from models import dusty
from models.gans import dcgan_eqlr


def define_G(cfg):
    masker_type, backbone_type = cfg.model.gen.arch.split("/")

    if backbone_type.lower() == "dcgan_eqlr":
        G = dcgan_eqlr.Generator(
            in_ch=cfg.model.gen.in_ch,
            out_ch=dict(cfg.model.gen.out_ch),
            ch_base=cfg.model.gen.ch_base,
            ch_max=cfg.model.gen.ch_max,
            shape=cfg.model.gen.shape,
            ring=cfg.model.ring,
        )
    else:
        raise NotImplementedError

    if masker_type == "dusty1":
        G = dusty.DUSty1(
            backbone=G,
            tau=cfg.model.gen.tau,
            drop_const=cfg.model.gen.drop_const,
        )
    elif masker_type == "dusty2":
        G = dusty.DUSty2(
            backbone=G,
            tau=cfg.model.gen.tau,
            drop_const=cfg.model.gen.drop_const,
        )
    elif masker_type == "none":
        pass
    else:
        raise NotImplementedError
    return G


def define_D(cfg):
    if cfg.model.dis.arch.lower() == "dcgan_eqlr":
        D = dcgan_eqlr.Discriminator(
            in_ch=cfg.model.dis.in_ch,
            ch_base=cfg.model.dis.ch_base,
            ch_max=cfg.model.dis.ch_max,
            shape=cfg.model.dis.shape,
            ring=cfg.model.ring,
        )
    else:
        raise NotImplementedError
    return D
