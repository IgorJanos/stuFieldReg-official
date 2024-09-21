# Football Playing Field Registration Using Distance Maps

This is the official repository of the paper Football Playing Field Registration Using Distance Maps.
You can find our original paper at - ..TODO..



The development environment is prepared inside the docker container. You can mount your host 
data folder into the container into the mount point `/workspace/stuFieldReg/.scratch`.

Please, inspect the `.devcontainer/devcontainer.json` and `docker/run.sh` files for more information.

This repository also contains parts of code refactored from and based on other github repositories. 
We wish to thank the authors for making their code and models publicly available:
 - [https://github.com/ericsujw/KpSFR](https://github.com/ericsujw/KpSFR)
 - [https://github.com/MM4SPA/tvcalib](https://github.com/MM4SPA/tvcalib)



## Datasets

In our work we use 3 datasets. You can download them at the following addresses:
 - Calib360 - [https://github.com/IgorJanos/stuCalib360](https://github.com/IgorJanos/stuCalib360)
 - WorldCup 2014 - [https://nhoma.github.io](https://nhoma.github.io)
 - TS WorldCup - [https://github.com/ericsujw/KpSFR](https://github.com/ericsujw/KpSFR)

## Pre-trained Model

You can download our best pre-trained models from:

 - [fieldreg-experiments.tar.gz](https://vggnas.fiit.stuba.sk/download/janos/fieldreg/fieldreg-experiments.tar.gz) 3.5 GB

Extract the content of the archive into the `/workspace/stuFieldReg/.scratch` folder
inside the container. The inference script expects to find the models 
in `/workspace/stuFieldReg/.scratch/experiments`.

## License

This work is licensed under MIT License. See [LICENSE](./LICENSE) for more details.

## Citing

If you find our work useful in your research, please consider citing:

```
TODO.
```

