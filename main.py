from model import SegNet
from data import batch_generator

net = SegNet()
net.fit_generator(batch_generator(), samples_per_epoch=13434, nb_epoch=5)