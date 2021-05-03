import trainer
from build_model import build_srresnet, build_disc

def pretrain():
    g = build_srresnet()
    gtrainer =  trainer.GPretrain(g)
    gtrainer.train()

def srgan_train():
    g = build_srresnet()
    d = build_disc()
    srgan_trainer = trainer.SRGANTrainer(g,d)
    srgan_trainer.train()
if __name__ == '__main__':
    srgan_train()

    